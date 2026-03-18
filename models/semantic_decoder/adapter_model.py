"""
adapter_model.py — Core model, data utilities, and helpers for adapter-based S2ST.

ARCHITECTURE:
  Hindi semantic units (discrete IDs)
       │
       ▼
  ┌─────────────────┐
  │ SpeechAdapter    │  Dedicated embedding (500 × 256) + MLP (256 → 2048)
  │ (TRAINABLE)      │  Lives in its own space, then projects to LLM space
  └────────┬────────┘
           │ embeddings that "look like text" to the LLM
           ▼
  ┌─────────────────┐
  │ AdapterLLM       │  Routes: speech tokens → adapter, text tokens → embed_tokens
  │ Qwen2.5-3B      │  Output: autoregressive Maithili semantic tokens via lm_head
  │ (LoRA in S2)    │
  └─────────────────┘

KEY DESIGN DECISIONS:
  1. Speech tokens BEFORE <SEP> → routed through adapter (Hindi input)
  2. Speech tokens AFTER <SEP> → routed through embed_tokens (Maithili target, teacher forcing)
  3. Text/control tokens → always through embed_tokens
  4. Output side → standard vocab extension + trained lm_head rows (same as approach 1)
"""

from __future__ import annotations

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple

import yaml
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    Trainer, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

IGNORE_INDEX = -100


# ===================================================================
# Logging
# ===================================================================
def log(msg: str):
    print(msg, flush=True)


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


# ===================================================================
# Speech Adapter
# ===================================================================
class SpeechAdapter(nn.Module):
    """
    Dedicated embedding table + MLP projector for semantic speech units.

    WHY a separate embedding instead of using LLM's embed_tokens:
      - The LLM's 150K-token embedding space is optimized for text.
        500 speech tokens crammed in there occupy a tiny, constrained corner.
      - A dedicated 256-dim space lets speech units organize by acoustic/
        linguistic similarity without interference from text geometry.
      - The MLP then learns a nonlinear mapping from this speech space
        into the LLM's 2048-dim hidden space, which is far more expressive
        than a single embedding lookup.

    Architecture:
      speech unit ID → Embedding(500, 256) → Linear(256, 1024) → GELU
                     → Dropout → Linear(1024, 2048) → LayerNorm → output
    """

    def __init__(self, num_units: int, speech_dim: int, llm_dim: int,
                 expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_units = num_units
        self.speech_dim = speech_dim
        self.llm_dim = llm_dim

        self.embed = nn.Embedding(num_units, speech_dim)
        self.proj = nn.Sequential(
            nn.Linear(speech_dim, speech_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(speech_dim * expansion, llm_dim),
            nn.LayerNorm(llm_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, local_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            local_ids: [B, T] — 0-indexed unit IDs (0 to num_units-1)
        Returns:
            [B, T, llm_dim] — projected embeddings in LLM space
        """
        return self.proj(self.embed(local_ids))


# ===================================================================
# Adapter-LLM Wrapper
# ===================================================================
class AdapterLLM(nn.Module):
    """
    Wraps LLM + SpeechAdapter. Routes tokens based on speech_mask:
      - speech_mask=True  → SpeechAdapter (Hindi input semantic tokens)
      - speech_mask=False → LLM embed_tokens (text, control, Maithili tokens)

    The LLM sees a unified sequence of embeddings and doesn't know which
    came from the adapter vs embed_tokens. It processes them identically
    through its transformer layers.
    """

    def __init__(self, llm: nn.Module, speech_adapter: SpeechAdapter,
                 sem_start_id: int, num_sem_tokens: int):
        super().__init__()
        self.llm = llm
        self.speech_adapter = speech_adapter
        self.sem_start = sem_start_id
        self.num_sem = num_sem_tokens

    @property
    def config(self):
        return self.llm.config

    @property
    def device(self):
        return next(self.parameters()).device

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                speech_mask=None, task=None, **kwargs):
        """
        Forward pass with speech routing.

        1. Get combined embeddings (adapter for speech, embed_tokens for rest)
        2. Feed to LLM via inputs_embeds (bypasses embed_tokens lookup)
        """
        combined = self._get_combined_embeds(input_ids, speech_mask)

        return self.llm(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _get_combined_embeds(self, input_ids, speech_mask):
        """Compute combined embeddings: adapter for speech, embed_tokens for rest."""
        text_embeds = self.get_input_embeddings()(input_ids)
        if speech_mask is not None and speech_mask.any():
            local_ids = (input_ids - self.sem_start).clamp(0, self.num_sem - 1)
            speech_embeds = self.speech_adapter(local_ids)
            mask_3d = speech_mask.unsqueeze(-1).to(dtype=speech_embeds.dtype)
            return speech_embeds * mask_3d + text_embeds * (1.0 - mask_3d)
        return text_embeds

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, speech_mask=None,
                 max_new_tokens=256, do_sample=False, temperature=1.0,
                 top_p=0.9, pad_token_id=None, eos_token_id=None, **kwargs):
        """
        Manual autoregressive generation.

        Why not use HF's generate():
          - HF generate + PEFT + inputs_embeds has known compatibility issues
          - When inputs_embeds is passed without input_ids, output alignment breaks
          - Manual loop gives us full control over the adapter→KV-cache→decode pipeline

        Steps:
          1. Compute combined embeddings (adapter + embed_tokens)
          2. Forward pass to fill KV cache
          3. Greedy/sample first token from logits
          4. Loop: feed each new token through embed_tokens (not adapter),
             use KV cache for efficiency
          5. Return [input_ids, generated_tokens]
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Enable KV cache for generation (disabled during training for grad ckpt)
        orig_use_cache = self.llm.config.use_cache
        self.llm.config.use_cache = True

        # Step 1: combined embeddings for the full input prefix
        combined = self._get_combined_embeds(input_ids, speech_mask)

        # Step 2: forward pass → fills KV cache
        outputs = self.llm(
            inputs_embeds=combined,
            attention_mask=attention_mask,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]  # [B, vocab]

        # Step 3: first token
        if do_sample and temperature > 0:
            probs = torch.softmax(next_logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cum_probs = sorted_probs.cumsum(dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = [next_token]  # each is [B, 1]
        kv_len = combined.shape[1]  # length already in the KV cache

        # Step 4: autoregressive loop
        for step in range(max_new_tokens - 1):
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            kv_len += 1
            # Attention mask grows by 1 each step
            cur_attn = torch.ones(batch_size, kv_len, device=device, dtype=torch.long)

            # New token goes through embed_tokens (standard LLM path, not adapter)
            outputs = self.llm(
                input_ids=next_token,
                past_key_values=past_kv,
                attention_mask=cur_attn,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(descending=True)
                    cum_probs = sorted_probs.cumsum(dim=-1)
                    mask = cum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(next_token)

        # Restore training setting
        self.llm.config.use_cache = orig_use_cache

        # Return full sequence: [input_ids, generated_tokens]
        gen_ids = torch.cat(generated, dim=-1)  # [B, gen_len]
        return torch.cat([input_ids, gen_ids], dim=-1)  # [B, inp_len + gen_len]


# ===================================================================
# Token ID Helpers
# ===================================================================
def get_sem_token_range(tokenizer) -> Tuple[int, int, int]:
    """
    Find the ID range of <SEM_*> tokens in the extended tokenizer.
    Returns: (sem_start_id, sem_end_id, count)
    """
    added = tokenizer.get_added_vocab()
    sem_ids = sorted(v for k, v in added.items() if k.startswith("<SEM_"))
    if not sem_ids:
        raise ValueError("No <SEM_*> tokens found in tokenizer")
    return sem_ids[0], sem_ids[-1], len(sem_ids)


def get_new_token_ids(tokenizer) -> List[int]:
    """All token IDs that were added during tokenizer extension."""
    added = tokenizer.get_added_vocab()
    ids = sorted(set(added.values()))
    if tokenizer.pad_token_id is not None and tokenizer.pad_token in added:
        ids = sorted(set(ids) | {tokenizer.pad_token_id})
    return ids


def get_sep_id(tokenizer) -> int:
    """Get the token ID for <SEP>."""
    sep_id = tokenizer.convert_tokens_to_ids("<SEP>")
    if sep_id == tokenizer.unk_token_id:
        raise ValueError("<SEP> token not found in tokenizer")
    return sep_id


# ===================================================================
# Speech Mask Computation
# ===================================================================
def compute_speech_mask(input_ids: torch.Tensor, sem_start: int,
                        sem_end: int, sep_id: int) -> torch.Tensor:
    """
    Compute boolean mask: True for semantic tokens BEFORE the first <SEP>.

    This distinguishes:
      - Hindi semantic tokens (before SEP) → True → routed to adapter
      - Maithili semantic tokens (after SEP) → False → routed to embed_tokens
      - Text/control tokens → False → routed to embed_tokens

    Args:
        input_ids: [B, T] token IDs
        sem_start: first semantic token ID (e.g., ID of <SEM_0>)
        sem_end: last semantic token ID (e.g., ID of <SEM_499>)
        sep_id: token ID of <SEP>
    Returns:
        [B, T] boolean tensor
    """
    is_sem = (input_ids >= sem_start) & (input_ids <= sem_end)
    is_sep = (input_ids == sep_id)
    # Cumulative sum of SEP: 0 before first SEP, ≥1 at and after SEP
    after_sep = is_sep.cumsum(dim=-1) > 0
    return is_sem & ~after_sep


# ===================================================================
# Embedding Helpers (for output side — same as approach 1)
# ===================================================================
def mean_init_new_embeddings(embedding_layer, original_vocab_size: int,
                             new_ids: List[int], name: str):
    """Initialize new token rows to mean(original) + noise."""
    with torch.no_grad():
        w = embedding_layer.weight
        mean = w[:original_vocab_size].float().mean(dim=0)
        for tid in new_ids:
            w[tid] = (mean + torch.randn_like(mean) * 0.02).to(w.dtype)
    norm = embedding_layer.weight[new_ids].float().norm(dim=1).mean().item()
    log(f"  {name}: mean-init {len(new_ids)} rows (norm={norm:.4f})")


def install_grad_mask_hook(embedding_layer, trainable_ids: List[int], name: str):
    """Gradient hook: only allow updates to trainable_ids rows."""
    param = embedding_layer.weight
    param.requires_grad_(True)
    vocab_size = param.shape[0]
    id_set = set(trainable_ids)
    cache = {}

    def hook(grad):
        key = f"{grad.device}_{grad.dtype}"
        if key not in cache:
            m = torch.zeros(vocab_size, 1, device=grad.device, dtype=grad.dtype)
            m[torch.tensor(sorted(id_set), device=grad.device)] = 1.0
            cache[key] = m
        return grad * cache[key]

    param.register_hook(hook)
    log(f"  {name}: grad hook → {len(trainable_ids)}/{vocab_size} rows trainable")


# ===================================================================
# Data Loading
# ===================================================================
def load_jsonl_by_task(path: str, task_limits: Dict[str, int],
                       seed: int = 42) -> List[Dict]:
    """Load JSONL, sample per task, shuffle."""
    by_task: Dict[str, List] = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            t = ex.get("task", "UNKNOWN")
            by_task.setdefault(t, []).append(ex)

    log(f"  Available: { {k: len(v) for k, v in by_task.items()} }")
    rng = random.Random(seed)
    result = []
    for task, limit in task_limits.items():
        avail = by_task.get(task, [])
        if limit <= 0 or not avail:
            continue
        chosen = rng.sample(avail, min(limit, len(avail)))
        result.extend(chosen)
        log(f"  {task}: {len(chosen)}/{len(avail)}")
    rng.shuffle(result)
    log(f"  Total: {len(result)}")
    return result


def build_dataset(samples: List[Dict], max_seq_len: int,
                  sem_start: int, sem_end: int, sep_id: int) -> Dataset:
    """
    Build HF Dataset with precomputed speech_mask.
    speech_mask[i] = 1 if token i is a semantic token before SEP, else 0.
    """
    all_input_ids = []
    all_labels = []
    all_attn = []
    all_speech_mask = []
    all_task = []

    for s in samples:
        ids = s["input_ids"][:max_seq_len]
        lab = s["labels"][:max_seq_len]
        att = s["attention_mask"][:max_seq_len]

        # Compute speech mask for this sample
        sm = []
        seen_sep = False
        for tok_id in ids:
            if tok_id == sep_id:
                seen_sep = True
            if not seen_sep and sem_start <= tok_id <= sem_end:
                sm.append(1)
            else:
                sm.append(0)

        all_input_ids.append(ids)
        all_labels.append(lab)
        all_attn.append(att)
        all_speech_mask.append(sm)
        all_task.append(s.get("task", "UNKNOWN"))

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attn,
        "speech_mask": all_speech_mask,
        "task": all_task,
    })


# ===================================================================
# Collator
# ===================================================================
def make_collator(pad_id: int):
    """Dynamic padding collator that includes speech_mask."""

    def collate(batch: List[Dict]) -> Dict[str, Any]:
        max_len = max(len(x["input_ids"]) for x in batch)

        def pad(seq, val):
            return seq + [val] * (max_len - len(seq))

        result = {
            "input_ids":      torch.tensor([pad(x["input_ids"], pad_id) for x in batch], dtype=torch.long),
            "labels":         torch.tensor([pad(x["labels"], IGNORE_INDEX) for x in batch], dtype=torch.long),
            "attention_mask": torch.tensor([pad(x["attention_mask"], 0) for x in batch], dtype=torch.long),
            "speech_mask":    torch.tensor([pad(x["speech_mask"], 0) for x in batch], dtype=torch.bool),
        }
        if "task" in batch[0]:
            result["task"] = [x["task"] for x in batch]
        return result

    return collate


# ===================================================================
# Save / Load
# ===================================================================
def save_adapter(adapter: SpeechAdapter, save_dir: str, label: str = ""):
    """Save speech adapter weights."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "speech_adapter.pt")
    torch.save(adapter.state_dict(), path)
    log(f"  [SaveAdapter] {label} → {path}")


def load_adapter(adapter: SpeechAdapter, load_dir: str):
    """Load speech adapter weights."""
    path = os.path.join(load_dir, "speech_adapter.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"speech_adapter.pt not found in {load_dir}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    adapter.load_state_dict(state)
    log(f"  Adapter loaded from {path}")


def save_embed_weights(model, save_dir: str, label: str = ""):
    """Save embed_tokens + lm_head weights for new tokens."""
    os.makedirs(save_dir, exist_ok=True)
    # Navigate through wrapper
    if hasattr(model, 'llm'):
        llm = model.llm
    else:
        llm = model
    inp = llm.get_input_embeddings()
    out = llm.get_output_embeddings()
    state = {
        "input_embeddings": inp.weight.data.cpu().clone(),
        "output_embeddings": out.weight.data.cpu().clone(),
    }
    path = os.path.join(save_dir, "embed_weights.pt")
    torch.save(state, path)
    log(f"  [SaveEmbeddings] {label} → {path}")


def load_embed_weights(model, load_dir: str):
    """Restore embed + lm_head weights."""
    path = os.path.join(load_dir, "embed_weights.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"embed_weights.pt not found in {load_dir}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if hasattr(model, 'llm'):
        llm = model.llm
    else:
        llm = model
    llm.get_input_embeddings().weight.data.copy_(
        state["input_embeddings"].to(llm.get_input_embeddings().weight.dtype))
    llm.get_output_embeddings().weight.data.copy_(
        state["output_embeddings"].to(llm.get_output_embeddings().weight.dtype))
    log(f"  Embeddings loaded from {path}")


# ===================================================================
# Callbacks
# ===================================================================
class SaveAllCallback(TrainerCallback):
    """Save adapter + embeddings at every checkpoint and end of training."""

    def on_save(self, args, state, control, model=None, **kwargs):
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if hasattr(model, 'speech_adapter'):
            save_adapter(model.speech_adapter, ckpt, f"step={state.global_step}")
        save_embed_weights(model, ckpt, f"step={state.global_step}")
        # If model has LoRA, save those too
        if hasattr(model, 'llm') and hasattr(model.llm, 'peft_config'):
            try:
                model.llm.save_pretrained(ckpt)
                log(f"  [SaveLoRA] step={state.global_step} → {ckpt}")
            except Exception:
                pass  # Stage 1 has no LoRA

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if hasattr(model, 'speech_adapter'):
            save_adapter(model.speech_adapter, args.output_dir, "final")
        save_embed_weights(model, args.output_dir, "final")
        if hasattr(model, 'llm') and hasattr(model.llm, 'peft_config'):
            try:
                model.llm.save_pretrained(args.output_dir)
                log(f"  [SaveLoRA] final → {args.output_dir}")
            except Exception:
                pass


# ===================================================================
# Task-Weighted Trainer
# ===================================================================
class AdapterTrainer(Trainer):
    """
    Custom Trainer for adapter-based S2ST.
    Handles: task-weighted loss + multi-group optimizer.
    """

    def __init__(self, *args, task_weights: Optional[Dict[str, float]] = None,
                 lr_groups: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {}
        self.lr_groups = lr_groups or {}

        log(f"\n  Task weights: {self.task_weights}")
        log(f"  LR groups: {self.lr_groups}")

    def create_optimizer(self):
        """
        Create optimizer with separate LR for adapter, LoRA, and embeddings.

        Groups:
          - "adapter": speech_adapter params → adapter_lr
          - "embed":   embed_tokens/lm_head → embed_lr
          - "lora":    everything else with requires_grad → lora_lr (base)
        """
        adapter_params, embed_params, other_params = [], [], []
        no_decay = {"bias", "LayerNorm", "layer_norm", "layernorm", "rmsnorm"}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "speech_adapter" in name:
                adapter_params.append((name, param))
            elif "embed" in name or "lm_head" in name:
                embed_params.append((name, param))
            else:
                other_params.append((name, param))

        base_lr = self.args.learning_rate
        adapter_lr = self.lr_groups.get("adapter", base_lr)
        embed_lr = self.lr_groups.get("embed", base_lr * 5)
        lora_lr = self.lr_groups.get("lora", base_lr)
        wd = self.args.weight_decay

        groups = []

        def add_group(params_list, lr):
            decay, nodecay = [], []
            for n, p in params_list:
                if any(nd in n for nd in no_decay) or p.dim() < 2:
                    nodecay.append(p)
                else:
                    decay.append(p)
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": wd})
            if nodecay:
                groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})

        add_group(adapter_params, adapter_lr)
        add_group(embed_params, embed_lr)
        add_group(other_params, lora_lr)

        log(f"\n  Optimizer groups:")
        log(f"    adapter: {sum(p.numel() for _, p in adapter_params):,} params (lr={adapter_lr:.6f})")
        log(f"    embed:   {sum(p.numel() for _, p in embed_params):,} params (lr={embed_lr:.6f})")
        log(f"    lora:    {sum(p.numel() for _, p in other_params):,} params (lr={lora_lr:.6f})")

        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        tasks = inputs.pop("task", None)
        outputs = model(**inputs)

        if tasks is None or not self.task_weights:
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        logits = outputs.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=IGNORE_INDEX)
        per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != IGNORE_INDEX).float()
        bw = torch.tensor(
            [self.task_weights.get(t, 1.0) for t in tasks],
            device=per_token.device, dtype=per_token.dtype,
        )
        tw = mask * bw.unsqueeze(1)
        loss = (per_token * tw).sum() / (tw.sum() + 1e-8)
        return (loss, outputs) if return_outputs else loss


    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override HF default full-model saving.
        We do NOT save the full tied-weight LLM.
        SaveAllCallback handles adapter + LoRA + embeddings.
        """
        if output_dir is None:
            output_dir = self.args.output_dir

        os.makedirs(output_dir, exist_ok=True)
        # Do nothing else.
        # Prevents safetensors from trying to serialize full model.
        # log(f"  [Trainer] Skipping full-model save at {output_dir}")
        # if hasattr(self.model, "llm"):
        #     try:
        #         self.model.llm.save_pretrained(output_dir)
        #     except Exception:
        #         pass


# ===================================================================
# Model Builders
# ===================================================================
def build_model_stage1(cfg: Dict[str, Any], device: str = "cuda:0"):
    """
    Build model for Stage 1 (alignment):
      - LLM: loaded, vocab resized, COMPLETELY FROZEN
      - SpeechAdapter: trainable
      - embed_tokens: mean-initialized for new tokens (frozen in stage 1)
      - No LoRA

    Returns: (AdapterLLM, tokenizer, sem_start, sem_end, sep_id)
    """
    log("\n[Stage 1] Building model")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_dir"], trust_remote_code=True, use_fast=True)
    log(f"  Vocab: {len(tokenizer)}")

    # LLM
    log(f"  Loading {cfg['base_model']} (bf16)")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    llm.config.use_cache = False
    llm.config.pad_token_id = tokenizer.pad_token_id

    # Resize vocab
    orig_vocab = llm.get_input_embeddings().weight.shape[0]
    llm.resize_token_embeddings(len(tokenizer))
    log(f"  Vocab resized: {orig_vocab} → {len(tokenizer)}")

    # Mean-init new embeddings (for control tokens used in embed_tokens)
    new_ids = get_new_token_ids(tokenizer)
    mean_init_new_embeddings(llm.get_input_embeddings(), orig_vocab, new_ids, "embed_tokens")
    tied = llm.get_input_embeddings().weight.data_ptr() == llm.get_output_embeddings().weight.data_ptr()
    if not tied:
        mean_init_new_embeddings(llm.get_output_embeddings(), orig_vocab, new_ids, "lm_head")

    # FREEZE entire LLM
    for param in llm.parameters():
        param.requires_grad_(False)
    log("  LLM: FROZEN (all parameters)")

    # Enable gradient checkpointing for memory efficiency
    # (needed for backward through frozen LLM to reach adapter gradients)
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Speech adapter
    sem_start, sem_end, num_sem = get_sem_token_range(tokenizer)
    llm_dim = llm.config.hidden_size
    adapter = SpeechAdapter(
        num_units=num_sem,
        speech_dim=cfg.get("speech_dim", 256),
        llm_dim=llm_dim,
        expansion=cfg.get("adapter_expansion", 4),
        dropout=cfg.get("adapter_dropout", 0.1),
    )
    adapter.to(device=device, dtype=llm.dtype)
    log(f"  Adapter: {num_sem} units → {cfg.get('speech_dim', 256)}d → {llm_dim}d")
    log(f"  Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    sep_id = get_sep_id(tokenizer)
    log(f"  SEM range: [{sem_start}, {sem_end}] ({num_sem} tokens)")
    log(f"  SEP ID: {sep_id}")

    # Wrap
    model = AdapterLLM(llm, adapter, sem_start, num_sem)
    model.to(device)

    # Verify trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer, sem_start, sem_end, sep_id


def build_model_stage2(cfg: Dict[str, Any], stage1_dir: str, device: str = "cuda:0"):
    """
    Build model for Stage 2 (S2ST fine-tuning):
      - LLM: loaded, vocab resized, LoRA applied
      - SpeechAdapter: loaded from stage 1, continues training
      - embed_tokens + lm_head: gradient hooks for new token rows
      - Full multi-component training

    Returns: (AdapterLLM, tokenizer, sem_start, sem_end, sep_id)
    """
    log("\n[Stage 2] Building model")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["tokenizer_dir"], trust_remote_code=True, use_fast=True)

    # LLM
    log(f"  Loading {cfg['base_model']} (bf16)")
    llm = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16,
        device_map={"": device}, trust_remote_code=True)
    llm.config.use_cache = False
    llm.config.pad_token_id = tokenizer.pad_token_id

    # Resize vocab
    orig_vocab = llm.get_input_embeddings().weight.shape[0]
    llm.resize_token_embeddings(len(tokenizer))
    new_ids = get_new_token_ids(tokenizer)
    mean_init_new_embeddings(llm.get_input_embeddings(), orig_vocab, new_ids, "embed_tokens")
    tied = llm.get_input_embeddings().weight.data_ptr() == llm.get_output_embeddings().weight.data_ptr()
    if not tied:
        mean_init_new_embeddings(llm.get_output_embeddings(), orig_vocab, new_ids, "lm_head")

    # LoRA
    log("  Applying LoRA")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none", inference_mode=False)
    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    # Gradient hooks for embed_tokens + lm_head (new token rows only)
    log("  Installing gradient hooks for new tokens")
    inp_emb = llm.get_input_embeddings()
    out_emb = llm.get_output_embeddings()
    install_grad_mask_hook(inp_emb, new_ids, "embed_tokens")
    if inp_emb.weight.data_ptr() != out_emb.weight.data_ptr():
        install_grad_mask_hook(out_emb, new_ids, "lm_head")
    else:
        log("  lm_head: tied → single hook")

    # Gradient checkpointing
    llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # Speech adapter — load from stage 1
    sem_start, sem_end, num_sem = get_sem_token_range(tokenizer)
    llm_dim = llm.config.hidden_size
    adapter = SpeechAdapter(
        num_units=num_sem,
        speech_dim=cfg.get("speech_dim", 256),
        llm_dim=llm_dim,
        expansion=cfg.get("adapter_expansion", 4),
        dropout=cfg.get("adapter_dropout", 0.1),
    )
    adapter.to(device=device, dtype=llm.dtype)
    load_adapter(adapter, stage1_dir)
    log(f"  Adapter loaded from stage 1: {stage1_dir}")

    sep_id = get_sep_id(tokenizer)

    # Wrap
    model = AdapterLLM(llm, adapter, sem_start, num_sem)
    model.to(device)

    # Verify
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer, sem_start, sem_end, sep_id