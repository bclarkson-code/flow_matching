import enum
import os

import hydra
import torch
from cleanfid import fid
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

from flow_matching.checkpoint import find_latest_checkpoint
from flow_matching.config import Config, register_configs


class DatasetName(enum.Enum):
    CIFAR_10: str = "cifar_10"
    COCO_CAPTIONS: str = "coco_captions"


def load_dataset(dataset: DatasetName) -> datasets.VisionDataset:
    match dataset:
        case DatasetName.CIFAR_10:
            return datasets.CIFAR10(root="./data", train=True, download=True)
        case DatasetName.COCO_CAPTIONS:
            return datasets.CocoCaptions(root="./data", train=True, download=True)
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")


def prepare_real_images(config: Config, dataset: datasets.VisionDataset):
    """Download CIFAR-10 and resize to target resolution."""
    os.makedirs(config.evaluation.real_dir, exist_ok=True)

    print("Resizing CIFAR-10 images...")
    for idx, (image, _) in enumerate(tqdm(dataset)):
        # Resize to target size
        image_size = config.dataset.image_size
        image = image.resize((image_size, image_size), Image.BILINEAR)
        image.save(f"{config.evaluation.real_dir}/{idx}.png")


def compute_real_stats(config: Config):
    """Compute FID statistics for the resized real images."""
    print("Computing statistics for real images...")
    fid.make_custom_stats(
        config.evaluation.stats_name, config.evaluation.real_dir, mode="clean"
    )


def generate_images(
    model, config: Config, dataset: datasets.VisionDataset, device="cuda"
):
    os.makedirs(config.evaluation.gen_dir, exist_ok=True)

    all_prompts = []
    for class_name in dataset.CLASSES:
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


def compute_fid(config: Config, dataset_name: DatasetName):
    """Compute FID score against custom stats."""
    print("Computing FID...")

    dataset = load_dataset(dataset_name)
    if (
        not os.path.exists(config.evaluation.real_dir)
        or len(os.listdir(config.evaluation.real_dir)) == 0
    ):
        prepare_real_images(config, dataset)
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

    if not fid.test_stats_exists(config.evaluation.stats_name, mode="clean"):
        compute_real_stats(config)
    generate_images(model, config, device)

    score = compute_fid(config)
    print(f"\nFID Score: {score:.2f}")
    return score


if __name__ == "__main__":
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)
    breakpoint()
    print(dir(ds))
