import os

import hydra
import torch
from cleanfid import fid
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

from src.flow_matching.checkpoint import find_latest_checkpoint
from src.flow_matching.config import Config, register_configs
from src.flow_matching.model import DiffusionTransformer, TextEmbedder


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_model_from_checkpoint(
    checkpoint_path: str,
    config: Config | None = None,
    device: torch.device | str = "cuda",
) -> DiffusionTransformer:
    if isinstance(device, str):
        device = torch.device(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config is None:
        config = checkpoint["config"]

    text_embedder = TextEmbedder(config)
    for parameter in text_embedder.parameters():
        parameter.requires_grad = False

    model = DiffusionTransformer(config)
    model = torch.compile(model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.text_embedder = text_embedder
    model = model.to(device)
    model.eval()

    step = checkpoint.get("step", "unknown")
    print(f"Loaded model from step {step}")

    return model


def prepare_real_images(config: Config):
    """Download CIFAR-10 and resize to target resolution."""
    os.makedirs(config.evaluation.real_dir, exist_ok=True)

    # Load CIFAR-10
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    print("Resizing CIFAR-10 images...")
    for idx, (image, _) in enumerate(tqdm(dataset)):
        # Resize to target size
        image_size = config.dataset.image_size
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image.save(f"{config.evaluation.real_dir}/{idx}.png")


def compute_real_stats(config: Config):
    """Compute FID statistics for the resized real images."""
    with torch.profiler.record_function("load_data"):
        batch = next(dataset)
    print("Computing statistics for real images...")
    fid.make_custom_stats(
        config.evaluation.stats_name, config.evaluation.real_dir, mode="clean"
    )


def generate_images(model, config: Config, device="cuda"):
    os.makedirs(config.evaluation.gen_dir, exist_ok=True)

    all_prompts = []
    for class_idx, class_name in enumerate(CIFAR10_CLASSES):
        prompt = f"a photo of a {class_name}"
        all_prompts.extend([prompt] * config.evaluation.images_per_class)

    total_images = len(all_prompts)
    print(f"Generating {total_images} images...")

    img_idx = 0
    batch_size = config.evaluation.batch_size
    for batch_start in tqdm(range(0, total_images, batch_size)):
        batch_end = min(batch_start + batch_size, total_images)
        batch_prompts = all_prompts[batch_start:batch_end]

        images = model.generate_images(batch_prompts, device=device)

        for image in images:
            image = image.cpu()
            image = (image * 255).clamp(0, 255).byte()
            image = Image.fromarray(image.permute(1, 2, 0).numpy())

            image_size = config.dataset.image_size
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.BILINEAR)

            image.save(f"{config.evaluation.gen_dir}/{img_idx}.png")
            img_idx += 1


def compute_fid(config: Config):
    """Compute FID score against custom stats."""
    print("Computing FID...")
    score = fid.compute_fid(
        config.evaluation.gen_dir,
        config.evaluation.real_dir,
        dataset_name=config.evaluation.stats_name,
        mode="clean",
        num_workers=0,
        use_dataparallel=False,
        device=torch.device("cuda:0"),
    )
    return score


register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    """Main evaluation function using Hydra for configuration management.

    Args:
        cfg: Hydra DictConfig containing all configuration parameters

    Usage:
        # Use default config with latest checkpoint
        python evaluate.py

        # Use a specific experiment config
        python evaluate.py +experiment=debug
        python evaluate.py +experiment=full_scale

        # Override specific parameters
        python evaluate.py evaluation.batch_size=128
        python evaluate.py evaluation.images_per_class=100

        # Specify a checkpoint path
        python evaluate.py checkpoint_path=checkpoints/step_10000.pt

        # Combine experiment with overrides
        python evaluate.py +experiment=debug evaluation.batch_size=64
    """
    config: Config = OmegaConf.to_object(cfg)  # type: ignore
    device = cfg.get("device", "cuda")

    checkpoint_path = find_latest_checkpoint(config.checkpoint.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {config.checkpoint.checkpoint_dir}"
        )

    model = load_model_from_checkpoint(checkpoint_path, device=device, config=config)

    if (
        not os.path.exists(config.evaluation.real_dir)
        or len(os.listdir(config.evaluation.real_dir)) == 0
    ):
        prepare_real_images(config)

    if not fid.test_stats_exists(config.evaluation.stats_name, mode="clean"):
        compute_real_stats(config)
    generate_images(model, config, device)

    score = compute_fid(config)
    print(f"\nFID Score: {score:.2f}")
    return score


if __name__ == "__main__":
    main()
