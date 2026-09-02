from dataclasses import dataclass, field

@dataclass
class ModelConfig:

    # Parameters
    context_length: int = 512
    vocab_size: int = 50304 # Actual gpt2 uses 50257, but 50304 is the nearest power of 2
    n_blocks: int = 6
    n_head: int = 6
    n_embd: int = 768

    # Optimizations
    use_flashattn: bool = True
    use_compile: bool = True

@dataclass
class TrainingConfig:

    data_root = "data_mixture_2.5B"
    total_batch_size: int = 524288
    micro_batch_size: int = 16
    max_lr: float = 6e-4
    min_lr: int = max_lr * 0.1
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.1
    val_every: int = 100
    checkpoint_every: int = 500
    eval_every: int = 250
    generate_every: int = 250
    num_return_sequences: int = 4
    max_length: int = 32

