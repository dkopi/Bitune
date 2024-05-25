from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple


@dataclass
class ThinkArguments:
    wandb: Optional[int] = field(
        default=0,
    )
    wandb_project: Optional[str] = field(
        default="bitune",
    )
    wandb_name: Optional[str] = field(
        default="default",
    )
    wandb_group: Optional[str] = field(
        default="default",
    )
    save_dir: Optional[str] = field(
        default=None,
    )
    seed: Optional[int] = field(
        default=42,
    )
    model_path: Optional[str] = field(
        default="google/gemma-2b",
    )
    dataset: Optional[str] = field(
        default="ultrafeedback",
    )
    train_batch_size: Optional[int] = field(
        default=128,
    )
    eval_batch_size: Optional[int] = field(
        default=128,
    )
    max_epochs: Optional[int] = field(
        default=1,
    )
    seq_length: Optional[int] = field(
        default=32,
    )
    beta1: Optional[float] = field(
        default=0.9,
    )
    beta2: Optional[float] = field(
        default=0.999,
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear",
    )
    num_warmup_steps: Optional[int] = field(
        default=1000,
    )
    max_train_steps: Optional[int] = field(
        default=10000,
    )
    max_eval_steps: Optional[int] = field(
        default=1000,
    )
    eval_freq: Optional[int] = field(
        default=1000,
    )
    learning_rate: Optional[float] = field(
        default=1e-3,
    )
    weight_decay: Optional[float] = field(
        default=0.0,
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1,
    )
    clip: Optional[float] = field(
        default=1.0,
    )
    running_loss_window: Optional[int] = field(
        default=10,
    )
    model_type: Optional[int] = field(
        default=0,
    )
    think_tokens: Optional[int] = field(
        default=0,
    )
    think_pos: Optional[int] = field(
        default=0,
    )
    attention_type: Optional[int] = field(
        default=0,
    )
    loss_masking: Optional[int] = field(
        default=0,
    )
    gradient_checkpointing: Optional[int] = field(
        default=0,
    )
    ft_type: Optional[int] = field(
        default=0,
    )
    think_lr: Optional[float] = field(
        default=0.0,
    )
    think_type: Optional[int] = field(
        default=0,
    )
    attn_type: Optional[int] = field(
        default=0,
    )
    skip_training: Optional[int] = field(
        default=0,
    )
    proj_type: Optional[int] = field(
        default=0,
    )
    n_pass: Optional[int] = field(
        default=0,
    )
    proj_idx: Optional[int] = field(
        default=0,
    )
    proj_channels: Optional[int] = field(
        default=512,
    )
    proj_bias: Optional[int] = field(
        default=0,
    )
    pass_type: Optional[int] = field(
        default=0,
    )
    rank: Optional[int] = field(
        default=8,
    )
    method_name: Optional[str] = field(
        default="",
    )
    token_init: Optional[int] = field(
        default=0,
    )
    bf16: Optional[int] = field(
        default=0,
    )
    skip_bidir: Optional[int] = field(
        default=0,
    )
    ablation: Optional[int] = field(
        default=0,
    )
    s_init: Optional[float] = field(
        default=0.01,
    )


def update_config(config, args):
    # put all args into config
    config.update(vars(args))
    return config
