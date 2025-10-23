from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
import torch
import torchvision
from PIL import Image
import time
from tqdm import tqdm
from config import Config


def preprocess_and_resize(row: dict, target_size: int = 64) -> dict:
    """Resize image to target size and keep metadata"""
    image = row["jpg"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((target_size, target_size), Image.LANCZOS)

    return {"jpg": image, "json": row["json"]}


def download_and_save_in_chunks(
    dataset_name: str,
    output_path: str,
    target_size: int = 64,
    split: str = "train",
    chunk_size: int = 10_000,
):
    """Download and save in chunks to avoid memory issues"""
    print(f"Loading {dataset_name} in streaming mode...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    chunk_idx = 0
    samples = []

    for idx, sample in enumerate(tqdm(dataset)):
        resized = preprocess_and_resize(sample, target_size)
        samples.append(resized)

        if (idx + 1) % chunk_size == 0:
            chunk_dataset = Dataset.from_list(samples)
            chunk_path = f"{output_path}_chunk_{chunk_idx}"
            chunk_dataset.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_idx} ({len(samples)} samples)")

            chunk_idx += 1
            samples = []

    if samples:
        chunk_dataset = Dataset.from_list(samples)
        chunk_path = f"{output_path}_chunk_{chunk_idx}"
        chunk_dataset.save_to_disk(chunk_path)
        print(f"Saved final chunk {chunk_idx} ({len(samples)} samples)")

    print("Concatenating chunks...")

    chunks = [load_from_disk(f"{output_path}_chunk_{i}") for i in range(chunk_idx + 1)]
    final_dataset = concatenate_datasets(chunks)
    final_dataset.save_to_disk(output_path)

    # for i in range(chunk_idx + 1):
    #     import shutil

    #     shutil.rmtree(f"{output_path}_chunk_{i}")

    print(f"Done! Saved {len(final_dataset)} samples to {output_path}")


def transform_image(
    row: dict[str, Image.Image | dict[str, str]],
) -> dict[str, torch.Tensor | str]:
    image = row["jpg"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float),
        ]
    )
    return {"image": transform(image), "text": row["json"]["prompt"]}


# Run this once
if __name__ == "__main__":
    config = Config()
    print("Loading from disk")
    dataset = load_from_disk(config.dataset_path)
    print("Loaded")
    print("Transforming")
    dataset = dataset.map(transform_image, keep_in_memory=True)
    print("transformed")
    dataset.save_to_disk("data/text-to-image-2M_64x64_transformed")

    # download_and_save_in_chunks(
    #     dataset_name=config.dataset_name,
    #     output_path=config.dataset_path,
    #     target_size=config.image_size,
    #     split="train",
    # )
