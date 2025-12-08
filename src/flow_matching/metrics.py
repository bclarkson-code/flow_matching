import datetime
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from flow_matching.config import Config
from flow_matching.eval_datasets import EvalDataset


def generate_images(
    model, config: Config, dataset: EvalDataset, device=torch.device("cpu")
) -> Path:
    img_idx = 0
    batch_size = config.evaluation.batch_size

    dataset.prepare()
    if dataset.prompts is None:
        raise ValueError("Dataset has not been prepared")

    total_images = len(dataset.prompts)
    now = datetime.datetime.now().isoformat()
    out_dir = Path(config.evaluation.gen_dir) / dataset.stats_name / now
    out_dir.mkdir(exist_ok=True, parents=True)

    for batch_start in tqdm(range(0, total_images, batch_size), desc="Generating"):
        batch_end = min(batch_start + batch_size, total_images)
        batch_prompts = dataset.prompts[batch_start:batch_end]

        images = model.generate_images(batch_prompts, device=device)

        for image in images:
            image = image.cpu()
            image = (image * 255).clamp(0, 255).byte()
            image = Image.fromarray(image.permute(1, 2, 0).numpy())

            image_size = config.dataset.image_size
            if image.size != (image_size, image_size):
                image = image.resize((image_size, image_size), Image.BILINEAR)

            image.save(out_dir / f"{img_idx}.png")
            img_idx += 1
    return out_dir


def compute_fid(
    model, config: Config, dataset: EvalDataset, device=torch.device("cpu")
) -> float:
    generated_dir = generate_images(model, config, dataset, device)
    return dataset.compute_fid(generated_dir, device)
