#!/usr/bin/env python3
"""
LoRA fine-tuning script for Chatterbox TTS.
Supports Bhojpuri and Maithili language adaptation.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Add parent directory for imports BEFORE any chatterbox imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Workaround for version import issue
import chatterbox
chatterbox.__version__ = "0.1.4"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
# from chatterbox.models.t3.lora_config import LoRAConfig, get_lora_config
from chatterbox.models.tokenizers import MTLTokenizer
from chatterbox.models.s3tokenizer import S3Tokenizer, S3_SR
from chatterbox.models.voice_encoder import VoiceEncoder

import librosa

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TTSDataset(Dataset):
    """Dataset for TTS training with text and speech tokens."""

    def __init__(
        self,
        manifest_path: str,
        tokenizer: MTLTokenizer,
        s3_tokenizer: S3Tokenizer,
        voice_encoder: VoiceEncoder,
        language_id: str,
        max_text_len: int = 512,
        max_speech_len: int = 2048,
        device: str = 'cpu'
    ):
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.language_id = language_id
        self.max_text_len = max_text_len
        self.max_speech_len = max_speech_len
        self.device = device

        # Load manifest
        with open(manifest_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get text tokens
        text = sample['text']
        text_tokens = self.tokenizer.text_to_tokens(
            text,
            language_id=self.language_id
        ).squeeze(0)

        # Truncate text if needed
        if len(text_tokens) > self.max_text_len:
            text_tokens = text_tokens[:self.max_text_len]

        # Get speech tokens
        if 'speech_tokens_path' in sample and sample['speech_tokens_path']:
            speech_tokens = torch.load(sample['speech_tokens_path'])
        else:
            # Tokenize audio on the fly
            audio_path = sample['audio_path']
            if audio_path.endswith('.npy'):
                audio = np.load(audio_path)
            else:
                audio, _ = librosa.load(audio_path, sr=S3_SR)

            with torch.no_grad():
                speech_tokens, _ = self.s3_tokenizer.forward([audio], max_len=self.max_speech_len)
                speech_tokens = speech_tokens.squeeze(0)

        # Truncate speech if needed
        if len(speech_tokens) > self.max_speech_len:
            speech_tokens = speech_tokens[:self.max_speech_len]

        # Get speaker embedding from audio
        audio_path = sample['audio_path']
        if audio_path.endswith('.npy'):
            audio = np.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=S3_SR)

        speaker_emb = torch.from_numpy(
            self.voice_encoder.embeds_from_wavs([audio], sample_rate=S3_SR)
        ).mean(dim=0)

        return {
            'text_tokens': text_tokens,
            'speech_tokens': speech_tokens,
            'speaker_emb': speaker_emb,
            'text': text,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Get max lengths
    max_text_len = max(len(item['text_tokens']) for item in batch)
    max_speech_len = max(len(item['speech_tokens']) for item in batch)

    batch_size = len(batch)

    # Initialize tensors
    text_tokens = torch.zeros(batch_size, max_text_len, dtype=torch.long)
    speech_tokens = torch.zeros(batch_size, max_speech_len, dtype=torch.long)
    text_lens = torch.zeros(batch_size, dtype=torch.long)
    speech_lens = torch.zeros(batch_size, dtype=torch.long)
    speaker_embs = torch.stack([item['speaker_emb'] for item in batch])

    # Fill tensors
    for i, item in enumerate(batch):
        text_len = len(item['text_tokens'])
        speech_len = len(item['speech_tokens'])

        text_tokens[i, :text_len] = item['text_tokens']
        speech_tokens[i, :speech_len] = item['speech_tokens']
        text_lens[i] = text_len
        speech_lens[i] = speech_len

    return {
        'text_tokens': text_tokens,
        'speech_tokens': speech_tokens,
        'text_lens': text_lens,
        'speech_lens': speech_lens,
        'speaker_embs': speaker_embs,
    }


class Trainer:
    """Trainer class for LoRA fine-tuning."""

    def __init__(
        self,
        model: T3,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device

        # Mixed precision
        self.scaler = GradScaler() if config.get('mixed_precision', True) else None

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Early stopping based on loss threshold
        self.no_improve_epochs = 0
        # self.early_stop_threshold = config.get('early_stop_threshold', 0.5)
        self.early_stop_patience = config.get('early_stop_patience', 3)
        # self.consecutive_low_loss_count = 0
        self.early_stopped = False

        # Checkpointing
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_text_loss = 0
        total_speech_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        grad_accum = self.config.get('grad_accum', 1)
        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            text_tokens = batch['text_tokens'].to(self.device)
            speech_tokens = batch['speech_tokens'].to(self.device)
            text_lens = batch['text_lens'].to(self.device)
            speech_lens = batch['speech_lens'].to(self.device)
            speaker_embs = batch['speaker_embs'].to(self.device)

            # Add start/stop tokens
            sot = self.model.hp.start_text_token
            eot = self.model.hp.stop_text_token
            sos = self.model.hp.start_speech_token
            eos = self.model.hp.stop_speech_token

            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            text_lens = text_lens + 2

            speech_tokens = F.pad(speech_tokens, (1, 0), value=sos)
            speech_tokens = F.pad(speech_tokens, (0, 1), value=eos)
            speech_lens = speech_lens + 2

            # Conditioning
            t3_cond = T3Cond(
                speaker_emb=speaker_embs.unsqueeze(1),
                emotion_adv=0.5 * torch.ones(len(text_tokens), 1, 1, device=self.device),
            )

            # --------------------
            # Forward + backward
            # --------------------
            if self.scaler:
                with autocast():
                    loss_text, loss_speech = self.model.loss(
                        t3_cond=t3_cond,
                        text_tokens=text_tokens,
                        text_token_lens=text_lens,
                        speech_tokens=speech_tokens,
                        speech_token_lens=speech_lens,
                    )
                    raw_loss = loss_text + loss_speech
                    loss = raw_loss / grad_accum

                self.scaler.scale(loss).backward()
            else:
                loss_text, loss_speech = self.model.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_lens,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_lens,
                )
                raw_loss = loss_text + loss_speech
                loss = raw_loss / grad_accum
                loss.backward()

            # --------------------
            # Optimizer step gate
            # --------------------
            do_step = (
                ((batch_idx + 1) % grad_accum == 0)
                or ((batch_idx + 1) == len(self.train_dataloader))
            )

            if do_step:
                if self.config.get('grad_clip', 0) > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['grad_clip']
                    )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            # --------------------
            # Metrics & logging
            # --------------------
            total_loss += raw_loss.item()
            total_text_loss += loss_text.item()
            total_speech_loss += loss_speech.item()
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{raw_loss.item():.4f}",
                'text': f"{loss_text.item():.4f}",
                'speech': f"{loss_speech.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            if self.global_step % self.config.get('log_interval', 100) == 0 and do_step:
                logger.info(
                    f"Step {self.global_step}: loss={raw_loss.item():.4f}, "
                    f"text={loss_text.item():.4f}, speech={loss_speech.item():.4f}"
                )

            # --------------------
            # Early stopping (raw loss only!)
            # --------------------
            # if raw_loss.item() < self.early_stop_threshold:
            #     self.consecutive_low_loss_count += 1
            #     if self.consecutive_low_loss_count >= self.early_stop_patience:
            #         logger.info(
            #             f"Early stopping triggered! Loss {raw_loss.item():.4f} < "
            #             f"{self.early_stop_threshold} for "
            #             f"{self.early_stop_patience} consecutive steps."
            #         )
            #         self.early_stopped = True
            #         early_stop_path = self.checkpoint_dir / "t3_full_early_stop.pt"
            #         torch.save(self.model.state_dict(), early_stop_path)
            #         logger.info(f"Early stop model saved to {early_stop_path}")
            #         break
            # else:
            #     self.consecutive_low_loss_count = 0

        avg_loss = total_loss / num_batches
        avg_text_loss = total_text_loss / num_batches
        avg_speech_loss = total_speech_loss / num_batches

        return {
            'loss': avg_loss,
            'text_loss': avg_text_loss,
            'speech_loss': avg_speech_loss,
        }

    @torch.no_grad()
    def validate(self):
        """Run validation."""
        if self.val_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0
        total_text_loss = 0
        total_speech_loss = 0
        num_batches = 0

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            text_tokens = batch['text_tokens'].to(self.device)
            speech_tokens = batch['speech_tokens'].to(self.device)
            text_lens = batch['text_lens'].to(self.device)
            speech_lens = batch['speech_lens'].to(self.device)
            speaker_embs = batch['speaker_embs'].to(self.device)

            # Add start/stop tokens
            sot = self.model.hp.start_text_token
            eot = self.model.hp.stop_text_token
            sos = self.model.hp.start_speech_token
            eos = self.model.hp.stop_speech_token

            text_tokens = F.pad(text_tokens, (1, 0), value=sot)
            text_tokens = F.pad(text_tokens, (0, 1), value=eot)
            text_lens = text_lens + 2

            speech_tokens = F.pad(speech_tokens, (1, 0), value=sos)
            speech_tokens = F.pad(speech_tokens, (0, 1), value=eos)
            speech_lens = speech_lens + 2

            t3_cond = T3Cond(
                speaker_emb=speaker_embs.unsqueeze(1),
                emotion_adv=0.5 * torch.ones(len(batch['text_tokens']), 1, 1, device=self.device),
            )

            loss_text, loss_speech = self.model.loss(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_lens,
            )

            total_loss += (loss_text + loss_speech).item()
            total_text_loss += loss_text.item()
            total_speech_loss += loss_speech.item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'text_loss': total_text_loss / num_batches,
            'speech_loss': total_speech_loss / num_batches,
        }

    def save_checkpoint(self, epoch: int, val_loss: float = None):
        """Save training checkpoint (full fine-tuning)."""

        # --------------------
        # Save full model weights
        # --------------------
        model_path = self.checkpoint_dir / f"t3_full_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path)

        # --------------------
        # Save training state
        # --------------------
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"trainer_state_epoch_{epoch}.pt"
        torch.save(state, checkpoint_path)

        logger.info(
            f"Saved checkpoint: model={model_path.name}, "
            f"trainer_state={checkpoint_path.name}"
        )

        # --------------------
        # Save best model
        # --------------------
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_model_path = self.checkpoint_dir / "t3_full_best.pt"
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved (val_loss={val_loss:.4f})")


    def train(self, num_epochs: int):
        """Run full training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Config: {self.config}")
        logger.info(
            f"Early stopping: patience={self.early_stop_patience} epochs (val loss)"
        )

        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch} - Train: loss={train_metrics['loss']:.4f}, "
                f"text={train_metrics['text_loss']:.4f}, "
                f"speech={train_metrics['speech_loss']:.4f}"
            )

            # Check if early stopping was triggered during this epoch
            if self.early_stopped:
                logger.info(f"Training stopped early at epoch {epoch}")
                break

            # Validate
            val_metrics = self.validate()
            if val_metrics:
                logger.info(
                    f"Epoch {epoch} - Val: loss={val_metrics['loss']:.4f}, "
                    f"text={val_metrics['text_loss']:.4f}, "
                    f"speech={val_metrics['speech_loss']:.4f}"
                )
                val_loss = val_metrics['loss']
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improve_epochs = 0
                else:
                    self.no_improve_epochs += 1

                if self.no_improve_epochs >= self.early_stop_patience:
                    logger.info(
                        f"Early stopping triggered: val_loss did not improve for "
                        f"{self.early_stop_patience} epochs."
                    )
                    break
            else:
                val_loss = train_metrics['loss']

            # Save checkpoint
            if epoch % self.config.get('save_interval', 1) == 0:
                self.save_checkpoint(epoch, val_loss)

        # Save final model
        # final_path = self.checkpoint_dir / "final_lora"
        # self.model.save_lora(str(final_path))

        final_path = self.checkpoint_dir / "t3_full_final.pt"
        torch.save(self.model.state_dict(), final_path)

        if self.early_stopped:
            logger.info(f"Training complete (early stopped). Final model saved to {final_path}")
        else:
            logger.info(f"Training complete. Final model saved to {final_path}")


def load_pretrained_model(checkpoint_dir: str, device: str = 'cuda') -> T3:
    """Load pretrained T3 model."""
    from safetensors.torch import load_file as load_safetensors

    checkpoint_dir = Path(checkpoint_dir)

    # Load T3 model
    t3 = T3(T3Config.multilingual())

    # Try different checkpoint names
    for ckpt_name in ['t3_mtl23ls_v2.safetensors', 't3_cfg.safetensors']:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            t3_state = load_safetensors(ckpt_path)
            if "model" in t3_state.keys():
                t3_state = t3_state["model"][0]
            t3.load_state_dict(t3_state)
            break
    else:
        raise FileNotFoundError(f"No T3 checkpoint found in {checkpoint_dir}")

    t3.to(device).eval()
    return t3


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Chatterbox TTS")

    # Data arguments
    parser.add_argument('--train-manifest', required=True, help='Training manifest path')
    parser.add_argument('--val-manifest', help='Validation manifest path')
    parser.add_argument('--language-id', required=True, help='Language code (bho, mai)')

    # Model arguments
    parser.add_argument('--checkpoint-dir', required=True, help='Pretrained model checkpoint directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for fine-tuned model')

    # LoRA arguments
    # parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    # parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    # parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--grad-accum', type=int, default=1, help='Gradient accumulation steps')

    # Early stopping arguments
    # parser.add_argument('--early-stop-threshold', type=float, default=0.5,
    #                     help='Loss threshold for early stopping (default: 0.5)')
    parser.add_argument('--early-stop-patience', type=int, default=20,
                        help='Number of consecutive steps below threshold to trigger early stop (default: 20)')

    # Other arguments
    parser.add_argument('--max-text-len', type=int, default=512, help='Max text length')
    parser.add_argument('--max-speech-len', type=int, default=2048, help='Max speech length')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--device', default='cuda', help='Device')
    # parser.add_argument('--mixed-precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    logger.info("Loading pretrained model...")
    model = load_pretrained_model(args.checkpoint_dir, args.device)

    # Apply LoRA
    # logger.info(f"Applying LoRA with rank={args.lora_rank}")
    # lora_config = LoRAConfig(
    #     r=args.lora_rank,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     language_id=args.language_id,
    # )
    # model.apply_lora(lora_config)
    # model.print_trainable_parameters()

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    # Load tokenizers
    logger.info("Loading tokenizers...")
    checkpoint_dir = Path(args.checkpoint_dir)

    tokenizer = MTLTokenizer(
        str(checkpoint_dir / "grapheme_mtl_merged_expanded_v1.json")
    )

    s3_tokenizer = S3Tokenizer()
    s3_tokenizer.to(args.device)

    voice_encoder = VoiceEncoder()

    # Never train these during T3 finetune
    for p in s3_tokenizer.parameters():
        p.requires_grad = False
    s3_tokenizer.eval()

    for p in voice_encoder.parameters():
        p.requires_grad = False
    voice_encoder.eval()


    ve_path = checkpoint_dir / "ve.pt"
    if ve_path.exists():
        voice_encoder.load_state_dict(torch.load(ve_path, weights_only=True))
    voice_encoder.to(args.device).eval()

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = TTSDataset(
        args.train_manifest,
        tokenizer,
        s3_tokenizer,
        voice_encoder,
        args.language_id,
        max_text_len=args.max_text_len,
        max_speech_len=args.max_speech_len,
        device=args.device
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_dataloader = None
    if args.val_manifest:
        val_dataset = TTSDataset(
            args.val_manifest,
            tokenizer,
            s3_tokenizer,
            voice_encoder,
            args.language_id,
            max_text_len=args.max_text_len,
            max_speech_len=args.max_speech_len,
            device=args.device
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            pin_memory=True
        )

    # Create optimizer
    # trainable_params = model.get_trainable_parameters()
    # optimizer = torch.optim.AdamW(
    #     trainable_params,
    #     lr=args.lr,
    #     weight_decay=args.weight_decay
    # )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)


    # Create scheduler
    steps_per_epoch = (len(train_dataloader) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps if total_steps > 0 else 0.1,
    )

    # Training config
    # config = {
    #     'checkpoint_dir': str(output_dir),
    #     'language_id': args.language_id,
    #     'lora_rank': args.lora_rank,
    #     'lora_alpha': args.lora_alpha,
    #     'batch_size': args.batch_size,
    #     'lr': args.lr,
    #     'epochs': args.epochs,
    #     'grad_clip': args.grad_clip,
    #     'mixed_precision': (not args.no_mixed_precision),
    #     'log_interval': 50,
    #     'save_interval': 1,
    #     'early_stop_threshold': args.early_stop_threshold,
    #     'early_stop_patience': args.early_stop_patience,
    # }

    config = {
        'checkpoint_dir': str(output_dir),
        'language_id': args.language_id,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'grad_clip': args.grad_clip,
        'grad_accum': args.grad_accum,
        'mixed_precision': (not args.no_mixed_precision),
        'log_interval': 50,
        'save_interval': 1,
        # 'early_stop_threshold': args.early_stop_threshold,
        'early_stop_patience': args.early_stop_patience,
    }

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device
    )

    trainer.train(args.epochs)

    logger.info("Training complete!")


if __name__ == '__main__':
    main()
