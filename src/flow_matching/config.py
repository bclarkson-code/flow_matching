import subprocess
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class ModelConfig:
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
class TrainingConfig:
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    batch_size: int = 128
    num_steps: int = 100_000
    warmup_ratio: float = 0.05
    lr_decay_ratio: float = 0.1
    gradient_clip_max_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class DatasetConfig:
    dataset_name: str = "jackyhate/text-to-image-2M"
    dataset_path: str = "data/text-to-image-2M_64x64/"
    dataset_pattern: str = "/mnt/storage/datasets/flow_matching/text-to-image-2M_64x64_preprocessed-%06d.tar"
    num_datapoints: int = 2_300_793
    train_dataset_pattern: str = "/mnt/storage/datasets/flow_matching/text-to-image-2M_64x64_preprocessed-{000001..000230}.tar"
    eval_dataset_pattern: str = "/mnt/storage/datasets/flow_matching/text-to-image-2M_64x64_preprocessed-000000.tar"
    eval_samples: int | None = 32
    dataset_size: int | None = None
    include_text_embedder: bool = False
    image_size: int = 64
    num_workers: int = 8
    prefetch_batches: int = 4
    shuffle_buffer_size: int = 100


@dataclass
class LoggingConfig:
    eval_every: int = 500
    num_inference_steps: int = 50
    log_every: int = 100
    use_wandb: bool = True
    wandb_project: str = "diffusion-transformer"
    num_images_to_upload: int = 32


@dataclass
class DistributedConfig:
    distributed: bool = True
    world_size: int = 2
    backend: str = "nccl"
    master_addr: str = "localhost"
    master_port: str = "12355"


@dataclass
class CheckpointConfig:
    save_checkpoints: bool = True
    checkpoint_freq: int = 500
    checkpoint_dir: str = "checkpoints"
    keep_recent_checkpoints: int = 5
    keep_checkpoint_every_n_steps: int = 2500


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda:0"
    resume_path: str | None = None

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)


def is_committed() -> bool:
    """
    Check that all the latest code changes are checked in
    and pushed to GitHub for reproducibility.
    Returns True if the working directory is clean (no uncommitted changes)
    and the local branch is up-to-date with the remote branch.
    """
    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip()
        if status_output:
            return False

        subprocess.check_call(
            ["git", "fetch"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
        ahead_behind = subprocess.check_output(
            [
                "git",
                "rev-list",
                "--left-right",
                "--count",
                f"origin/{branch}...{branch}",
            ],
            text=True,
        ).strip()

        behind, ahead = map(int, ahead_behind.split())
        return ahead == 0 and behind == 0

    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        return False
