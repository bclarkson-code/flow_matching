import os

import torch
from cleanfid import fid
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from src.flow_matching.config import Config


# CIFAR-10 class labels
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


def prepare_real_images(config: Config):
    """Download CIFAR-10 and resize to target resolution."""
    os.makedirs(config.evaluation.real_dir, exist_ok=True)

    # Load CIFAR-10
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    print("Resizing CIFAR-10 images...")
    for idx, (image, label) in enumerate(tqdm(dataset)):
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


def generate_images(model, config: Config, device="cuda"):
    """Generate images using batched inference."""
    os.makedirs(config.evaluation.gen_dir, exist_ok=True)

    # Build list of all prompts
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

        # Generate batch
        with torch.no_grad():
            images = model.generate(batch_prompts)  # Adjust to your model's API

        # Process and save each image
        for image in images:
            if isinstance(image, torch.Tensor):
                image = image.cpu()
                # Handle [-1, 1] or [0, 1] range
                if image.min() < 0:
                    image = (image + 1) / 2
                image = (image * 255).clamp(0, 255).byte()
                image = Image.fromarray(image.permute(1, 2, 0).numpy())

            # Ensure correct size
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
        dataset_name=config.evaluation.stats_name,
        mode="clean",
    )
    return score


def main(model, config: Config | None = None, device="cuda"):
    """
    Main evaluation function.

    Args:
        model: The model to evaluate
        config: Configuration object. If None, uses default Config()
        device: Device to use for generation
    """
    if config is None:
        config = Config()

    # Step 1: Prepare real images (only need to do once)
    if (
        not os.path.exists(config.evaluation.real_dir)
        or len(os.listdir(config.evaluation.real_dir)) == 0
    ):
        prepare_real_images(config)

    # Step 2: Compute stats for real images (only need to do once)
    # cleanfid caches stats, so this checks if they exist
    try:
        fid.compute_fid(
            config.evaluation.gen_dir,
            dataset_name=config.evaluation.stats_name,
            mode="clean",
        )
    except:
        compute_real_stats(config)

    # Step 3: Generate images
    generate_images(model, config, device)

    # Step 4: Compute FID
    score = compute_fid(config)
    print(f"\nFID Score: {score:.2f}")
    return score


# Usage:
# from src.flow_matching.config import Config
# config = Config()
# model = YourModel().to(config.device)
# main(model, config)
