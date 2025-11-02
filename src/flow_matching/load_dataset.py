import numpy as np
import torch
import torchvision
from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm
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
    dataset = load_from_disk(dataset_path)
    print("Loading text encoder...")
    text_embedder = TextEmbedder(config)
    image_embedder = ImageEmbedder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder = text_embedder.eval().to(device)
    image_embedder = image_embedder.eval().to(device)

    sink = ShardWriter(output_pattern, maxcount=samples_per_shard)
    to_tensor = torchvision.transforms.PILToTensor()

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image: Image.Image = sample["jpg"]  # pyright: ignore[reportAssignmentType]

        if image.mode != "RGB":
            image = image.convert("RGB")

        with torch.no_grad():
            text_embeds, attention_mask = text_embedder([sample["json"]["prompt"]])  # type: ignore
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


if __name__ == "__main__":
    config = FullScaleConfig()
    create_webdataset(
        dataset_path=config.dataset_path,
        output_pattern=config.dataset_pattern,
        config=config,
    )
