# Copyright (c) 2025
# LoRA configuration for Chatterbox TTS fine-tuning

import logging
from typing import Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning."""

    # LoRA hyperparameters
    r: int = 16  # Rank of the low-rank matrices
    lora_alpha: int = 32  # Scaling factor
    lora_dropout: float = 0.05  # Dropout probability for LoRA layers

    # Target modules in Llama for LoRA
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",  # Query projection
        "k_proj",  # Key projection
        "v_proj",  # Value projection
        "o_proj",  # Output projection
        "gate_proj",  # MLP gate projection
        "up_proj",  # MLP up projection
        "down_proj",  # MLP down projection
    ])

    # Additional configuration
    bias: str = "none"  # Bias training: "none", "all", "lora_only"
    task_type: str = "FEATURE_EXTRACTION"  # Task type for PEFT (not CAUSAL_LM since we use LlamaModel not LlamaForCausalLM)
    inference_mode: bool = False  # Set True for inference

    # Language-specific settings
    language_id: Optional[str] = None  # e.g., "bho" for Bhojpuri

    @classmethod
    def for_bhojpuri(cls, r: int = 16):
        """Create LoRA config optimized for Bhojpuri fine-tuning."""
        return cls(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=0.05,
            language_id="bho",
        )

    @classmethod
    def for_maithili(cls, r: int = 16):
        """Create LoRA config optimized for Maithili fine-tuning."""
        return cls(
            r=r,
            lora_alpha=r * 2,
            lora_dropout=0.05,
            language_id="mai",
        )

    @classmethod
    def minimal(cls):
        """Minimal LoRA config for quick experiments (fewer parameters)."""
        return cls(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],  # Only attention
            lora_dropout=0.1,
        )

    @classmethod
    def full(cls):
        """Full LoRA config for maximum adaptation capability."""
        return cls(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
        )

    def to_peft_config(self):
        """Convert to PEFT LoraConfig object."""
        try:
            from peft import LoraConfig as PeftLoraConfig

            return PeftLoraConfig(
                r=self.r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.bias,
                task_type=self.task_type,
                inference_mode=self.inference_mode,
            )
        except ImportError:
            raise ImportError(
                "PEFT library not installed. Install with: pip install peft"
            )


def get_lora_config(language: str = None, rank: int = 16) -> LoRAConfig:
    """
    Get appropriate LoRA configuration for a language.

    Args:
        language: Language code (e.g., "bho", "mai")
        rank: LoRA rank (8, 16, 32)

    Returns:
        LoRAConfig instance
    """
    if language == "bho":
        return LoRAConfig.for_bhojpuri(r=rank)
    elif language == "mai":
        return LoRAConfig.for_maithili(r=rank)
    else:
        config = LoRAConfig(r=rank, lora_alpha=rank * 2)
        config.language_id = language
        return config


def apply_lora_to_model(model, lora_config: LoRAConfig):
    """
    Apply LoRA adapters to a T3 model's Llama backbone.

    Args:
        model: T3 model instance
        lora_config: LoRAConfig instance

    Returns:
        Model with LoRA adapters applied
    """
    try:
        from peft import get_peft_model

        peft_config = lora_config.to_peft_config()

        # Apply LoRA to the Llama transformer backbone
        model.tfmr = get_peft_model(model.tfmr, peft_config)

        # Log parameter counts
        trainable_params = sum(
            p.numel() for p in model.tfmr.parameters() if p.requires_grad
        )
        total_params = sum(p.numel() for p in model.tfmr.parameters())

        logger.info(
            f"LoRA applied: {trainable_params:,} trainable params "
            f"({100 * trainable_params / total_params:.2f}% of {total_params:,} total)"
        )

        return model

    except ImportError:
        raise ImportError(
            "PEFT library not installed. Install with: pip install peft"
        )


def save_lora_weights(model, save_path: str):
    """
    Save only the LoRA adapter weights.

    Args:
        model: T3 model with LoRA applied
        save_path: Path to save the adapter weights
    """
    try:
        model.tfmr.save_pretrained(save_path)
        logger.info(f"LoRA weights saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save LoRA weights: {e}")
        raise


def load_lora_weights(model, load_path: str):
    """
    Load LoRA adapter weights into a model.

    Args:
        model: T3 model instance
        load_path: Path to the saved adapter weights

    Returns:
        Model with loaded LoRA weights
    """
    try:
        from peft import PeftModel

        # Set attention implementation to 'eager' before loading
        # This is required for output_attentions to work during inference
        if hasattr(model.tfmr, 'config'):
            model.tfmr.config._attn_implementation = 'eager'

        model.tfmr = PeftModel.from_pretrained(model.tfmr, load_path)

        # Set attention implementation on all possible config locations
        # PeftModel wraps the model in multiple layers
        if hasattr(model.tfmr, 'config'):
            model.tfmr.config._attn_implementation = 'eager'

        if hasattr(model.tfmr, 'base_model'):
            if hasattr(model.tfmr.base_model, 'config'):
                model.tfmr.base_model.config._attn_implementation = 'eager'
            if hasattr(model.tfmr.base_model, 'model') and hasattr(model.tfmr.base_model.model, 'config'):
                model.tfmr.base_model.model.config._attn_implementation = 'eager'

        logger.info(f"LoRA weights loaded from {load_path}")

        return model

    except ImportError:
        raise ImportError(
            "PEFT library not installed. Install with: pip install peft"
        )


def merge_lora_weights(model):
    """
    Merge LoRA weights into base model for faster inference.

    Args:
        model: T3 model with LoRA applied

    Returns:
        Model with merged weights (no longer has separate LoRA adapters)
    """
    try:
        model.tfmr = model.tfmr.merge_and_unload()
        logger.info("LoRA weights merged into base model")
        return model
    except Exception as e:
        logger.error(f"Failed to merge LoRA weights: {e}")
        raise


def print_trainable_parameters(model):
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable %: {100 * trainable_params / all_params:.4f}"
    )
