import multiprocessing
import os
import time

import numpy as np
import torch
import torchvision
from datasets import load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from webdataset import ShardWriter  # type: ignore

from flow_matching.config import Config, FullScaleConfig
from flow_matching.model import ImageEmbedder, TextEmbedder


def preprocess_and_resize(row: dict, target_size: int = 64) -> dict:
    """Resize image to target size and keep metadata"""
    image = row["jpg"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((target_size, target_size), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]

    return {"jpg": image, "json": row["json"]}


def create_webdataset(
    dataset_path: str,
    output_pattern: str,
    config: Config,
    samples_per_shard: int = 10000,
):
    """
    Save to WebDataset format (TAR archives)

    Args:
        output_pattern: e.g., "data/shard-%06d.tar" creates data/shard-000000.tar, etc.
        samples_per_shard: Number of samples per TAR file (10k is typical)
    """
    dataset = load_from_disk(dataset_path=dataset_path)
    print("Loading text encoder...")
    text_embedder = TextEmbedder(config)
    image_embedder = ImageEmbedder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder = text_embedder.eval().to(device)
    image_embedder = image_embedder.eval().to(device)

    sink = ShardWriter(output_pattern, maxcount=samples_per_shard)
    to_tensor = torchvision.transforms.PILToTensor()

    for idx in tqdm(range(10_000)):
        image: Image.Image = dataset["jpg"][idx]  # pyright: ignore[reportAssignmentType]

        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((config.image_size, config.image_size), Image.LANCZOS)  # pyright: ignore[reportAttributeAccessIssue]

        with torch.no_grad():
            text_embeds, attention_mask = text_embedder(
                [dataset["json"][idx]["prompt"]]
            )  # type: ignore
            text_embeds = text_embeds[0].cpu().numpy().astype(np.float16)
            attention_mask = attention_mask[0].cpu().numpy()

            image_tensor = to_tensor(image)
            image_tensor = image_tensor.float().to(device)
            image_tensor = image_tensor / 256  # [0,255] -> [0,1]
            image_tensor = (image_tensor * 2) - 1  # [0,1] -> [-1, 1]
            image_tensor = image_tensor.unsqueeze(0)
            latents = image_embedder.to_latent(image_tensor)
            latents = latents.cpu().numpy().astype(np.float16)

        sink.write(
            {
                "__key__": f"{idx:08d}",
                "latents.pyd": latents,
                "embeds.pyd": text_embeds,
                "mask.pyd": attention_mask,
            }
        )

    sink.close()


def process_sample(sample):
    print("start")
    image: Image.Image = sample["jpg"]  # pyright: ignore[reportAssignmentType]

    print("loaded image")
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((config.image_size, config.image_size), Image.LANCZOS)  # type: ignore
    print("resized image")
    prompt = sample["json"]["prompt"]  # type: ignore

    image_tensor = (
        torchvision.transforms.PILToTensor()(image).float() / 256
    )  # [0,255] -> [0,1]
    print("to_tensor")
    image_tensor = (image_tensor * 2) - 1  # [0,1] -> [-1, 1]
    return image_tensor, prompt


def create_webdataset_batched(
    output_pattern: str,
    config: Config,
    samples_per_shard: int = 10000,
    batch_size: int = 32,
):
    """
    Save to WebDataset format (TAR archives)

    Args:
        output_pattern: e.g., "data/shard-%06d.tar" creates data/shard-000000.tar, etc.
        samples_per_shard: Number of samples per TAR file (10k is typical)
        batch_size: Number of samples to process in parallel (default: 32)
    """
    dataset = load_dataset(config.dataset_name, split="train")
    dataset = iter(dataset)
    print("Loading encoders...")
    text_embedder = TextEmbedder(config)
    image_embedder = ImageEmbedder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder = text_embedder.eval().to(device)
    image_embedder = image_embedder.eval().to(device)

    sink = ShardWriter(output_pattern, maxcount=samples_per_shard)
    index = 0
    running = True
    pbar = tqdm(total=config.num_datapoints)

    while running:
        batch_samples = []
        for _ in range(batch_size):
            try:
                batch_samples.append(next(dataset))
            except StopIteration:
                running = False

        prompts = []
        image_tensors = []

        with multiprocessing.Pool(4) as pool:
            processed = pool.map(process_sample, batch_samples)

        for image_tensor, prompt in processed:
            prompts.append(prompt)  # type: ignore
            image_tensors.append(image_tensor)

        with torch.no_grad():
            text_embeds, attention_masks = text_embedder(prompts)  # type: ignore

            image_batch = torch.stack(image_tensors).to(device)
            latents_batch = image_embedder.to_latent(image_batch)

            text_embeds = text_embeds.cpu().numpy().astype(np.float16)
            attention_masks = attention_masks.cpu().numpy()
            latents_batch = latents_batch.cpu().numpy().astype(np.float16)

            for text_embed, attention_mask, latents in zip(
                text_embeds, attention_masks, latents_batch
            ):
                sink.write(
                    {
                        "__key__": f"{index:08d}",
                        "latents.pyd": latents,
                        "embeds.pyd": text_embed,
                        "mask.pyd": attention_mask,
                    }
                )
                pbar.update()
                index += 1
    pbar.close()
    sink.close()


if __name__ == "__main__":
    config = FullScaleConfig()
    start = time.time()
    create_webdataset_batched(
        output_pattern=config.dataset_pattern,
        config=config,
    )
    duration = time.time() - start
    print(f"Took: {duration:.5f}s")
