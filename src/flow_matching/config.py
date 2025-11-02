from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for DiffusionTransformer model architecture"""

    text_embed_model_string: str = "t5-small"
    image_embed_model_string: str = "stabilityai/sd-vae-ft-mse"
    embedding_dim: int = 768
    time_embedding_dim: int = 256
    text_embedding_dim: int = 512
    n_image_tokens: int = 64
    n_text_tokens: int = 128
    n_blocks: int = 12
    n_heads: int = 12
    expansion_factor: int = 4
    latent_channels: int = 4
    vae_scale_factor: float = 0.18215
    patch_embed_init: float = 0.02
    final_layer_init_gain: float = 0.02
    text_max_length: int = 128


@dataclass
class Config:
    model_name: str = "DiffusionTransformer"
    model: ModelConfig = field(default_factory=ModelConfig)

    learning_rate: float = 5e-4
    weight_decay: float = 1e-2
    batch_size: int = 128
    num_steps: int = 100_000
    num_repeats: int = 1_000_000
    warmup_ratio: float = 0.05
    lr_decay_ratio: float = 0.1
    shuffle_buffer_size = 100
    gradient_clip_max_norm: float = 5.0
    gradient_accumulation_steps: int = 1

    dataset_name: str = "jackyhate/text-to-image-2M"
    dataset_path: str = "data/text-to-image-2M_64x64/"
    dataset_pattern: str = "data/text-to-image-2M_64x64_preprocessed-%06d.tar"
    num_datapoints = 2_300_793
    train_dataset_pattern: str = (
        "data/text-to-image-2M_64x64_preprocessed-{000001..000230}.tar"
    )
    eval_dataset_pattern: str = "data/text-to-image-2M_64x64_preprocessed-000000.tar"
    eval_samples: int | None = 32
    dataset_size: int | None = None
    include_text_embedder: bool = False
    skip_first_n_samples: int = 0
    image_size: int = 64
    num_workers: int = 8
    prefetch_batches: int = 4

    eval_every: int = 500
    num_inference_steps: int = 50
    log_every: int = 100
    seed: int = 0

    use_wandb: bool = True
    wandb_project: str = "diffusion-transformer"
    wandb_watch_log: Literal["gradients", "parameters", "all"] | None = "all"
    wandb_watch_log_freq: int = 100
    num_images_to_upload: int = 32

    device: str = "cuda:0"

    distributed: bool = True
    world_size: int = 2
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"

    save_checkpoints: bool = True
    checkpoint_freq: int = 500
    checkpoint_dir: str = "checkpoints"
    keep_recent_checkpoints: int = 5
    keep_checkpoint_every_n_steps: int = 2500


@dataclass
class OverfitConfig(Config):
    """
    Check that we can overfit a small (4096) number of images
    """

    eval_every: int = 1000
    save_checkpoints: bool = False
    dataset_size: int | None = 4096
    batch_size: int = 128
    gradient_accumulation_steps: int = 2
    num_steps: int = 1_000
    learning_rate: float = 1e-4


@dataclass
class HParamTuningConfig(Config):
    """
    Run a relatively small scale run (5k steps) to dial in hyperparameters like
    learning rate
    """

    log_every: int = 1000
    eval_samples: int | None = 1024
    shuffle_buffer_size: int = 100_000
    save_checkpoints: bool = False
    dataset_size: int | None = None

    # effective batch size = 128 * 2 * 2 = 512
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    distributed: bool = False
    world_size: int = 1
    train_dataset_pattern: str = (
        "data/text-to-image-2M_64x64_preprocessed-{000001..000050}.tar"
    )

    num_steps: int = 5_000

    learning_rate: float = 1e-3
    gradient_clip_max_norm: float = 5.0


@dataclass
class DebugConfig(Config):
    """Config used for one off debugging"""

    batch_size: int = 128
    distributed: bool = False
    gradient_accumulation_steps: int = 2
    eval_samples: int | None = 128
    eval_every: int = 100
    dataset_size: int | None = 100_000
    save_checkpoints: bool = False
    num_workers: int = 0
    world_size: int = 1
    num_steps: int = 500
    use_wandb: bool = True


@dataclass
class StabilityConfig(Config):
    """
    Medium scale run to check that the model can actually train stably for a while
    """

    eval_every: int = 1000
    eval_samples: int | None = 1024
    num_images_to_upload: int = 100
    save_checkpoints: bool = True
    keep_checkpoint_every_n_steps: int = 10_000
    dataset_size: int | None = None

    # effective batch size = 128 * 2 * 2 = 512
    batch_size: int = 128
    gradient_accumulation_steps: int = 2
    world_size: int = 2

    # epochs = 30_000 * 576 / 2_000_000 ~= 8.5
    num_steps: int = 30_000

    # got this from small scale hparam tuning
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.05


@dataclass
class FullScaleConfig(Config):
    """
    Full scale run to train the best model I can on 2 3090's
    """

    eval_every: int = 1000
    eval_samples: int | None = 1024
    save_checkpoints: bool = True
    keep_checkpoint_every_n_steps: int = 50_000
    dataset_size: int | None = None
    # effective batch size = 96 * 3 * 2 = 576
    batch_size: int = 512
    gradient_accumulation_steps: int = 1
    distributed: bool = False
    world_size: int = 1

    # epochs = 300_000 * 576 / 2_000_000 ~= 85
    num_steps: int = 300_000

    # got this from small scale hparam tuning
    learning_rate: float = 5e-4
    warmup_ratio: float = 0.05


class ConfigType(Enum):
    DEFAULT = "default"
    OVERFIT = "overfit"
    HPARAM_TUNING = "hparam_tuning"
    DEBUG = "debug"
    STABILITY = "stability"
    FULL_SCALE = "full_scale"

    def to_config(self) -> Config:
        match self:
            case ConfigType.DEFAULT:
                return Config()
            case ConfigType.OVERFIT:
                return OverfitConfig()
            case ConfigType.HPARAM_TUNING:
                return HParamTuningConfig()
            case ConfigType.DEBUG:
                return DebugConfig()
            case ConfigType.STABILITY:
                return StabilityConfig()
            case ConfigType.FULL_SCALE:
                return FullScaleConfig()
