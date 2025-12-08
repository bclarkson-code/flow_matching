import abc
import pickle
from pathlib import Path
from typing import Iterable

import datasets as huggingface_datasets
import torch
from cleanfid import fid
from PIL import Image
from torchvision import datasets as torchvision_datasets
from tqdm import tqdm

from flow_matching.config import Config


class EvalDataset(abc.ABC):
    real_dir: Path
    stats_name: str
    image_size: int
    config: Config
    prompts: list[str] | None = None

    @abc.abstractmethod
    def prepare(self) -> None:
        pass

    def compute_real_stats(self):
        fid.make_custom_stats(
            self.stats_name,
            self.real_dir,
            mode="clean",
        )

    def have_precomputed_stats(self) -> bool:
        return fid.test_stats_exists(self.stats_name, mode="clean")

    def compute_fid(
        self, generated_dir: Path, device: torch.device = torch.device("cpu")
    ) -> float:
        if not Path(self.real_dir).exists:
            self.prepare()

        return fid.compute_fid(
            str(generated_dir),
            str(self.real_dir),
            dataset_name=self.stats_name,
            dataset_res=self.image_size,
            mode="clean",
            num_workers=0,
            use_dataparallel=False,
            device=device,
        )


class Cifar10(EvalDataset):
    def __init__(self, config: Config):
        self.dataset = torchvision_datasets.CIFAR10(
            root="./data", train=True, download=True
        )
        self.real_dir = Path(config.evaluation.real_dir) / "cifar_10"
        self.config = config
        self.image_size = config.dataset.image_size
        self.stats_name = f"cifar10_resized_{self.image_size}"
        self.prompts_path = self.real_dir / "prompts.pkl"

        if self.prompts_path.exists():
            with open(self.prompts_path, "rb") as f:
                self.prompts = pickle.load(f)

    def prepare(self) -> None:
        if self.real_dir.exists() and self.prompts is not None:
            return

        self.real_dir.mkdir(parents=True, exist_ok=True)
        prompts = []

        n_images = self.config.evaluation.n_eval_images
        for idx, (image, label) in enumerate(
            tqdm(self.dataset, desc="Preparing", total=n_images)
        ):
            if idx >= n_images:
                break
            image_size = self.image_size
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image.save(self.real_dir / f"{idx}.png")

            prompts.append(f"a photo of a {label}")

        self.prompts = prompts
        with open(self.prompts_path, "wb") as f:
            pickle.dump(self.prompts, f)


class COCOCaptions(EvalDataset):
    def __init__(self, config: Config):
        self.dataset = huggingface_datasets.load_dataset(
            "lmms-lab/COCO-Caption2017", split="val", streaming=True
        )

        self.dataset = self.dataset.shuffle(seed=0, buffer_size=10000)
        self.real_dir = Path(config.evaluation.real_dir) / "coco_captions"
        self.image_size = config.dataset.image_size
        self.stats_name = f"coco_captions_resized_{self.image_size}"
        self.prompts_path = self.real_dir / "prompts.pkl"
        self.config = config

        if self.prompts_path.exists():
            with open(self.prompts_path, "rb") as f:
                self.prompts = pickle.load(f)

    def prepare(self) -> None:
        if self.real_dir.exists() and self.prompts is not None:
            return

        self.real_dir.mkdir(parents=True, exist_ok=True)
        prompts = []

        n_images = self.config.evaluation.n_eval_images

        for idx, batch in enumerate(
            tqdm(self.dataset, desc="Preparing", total=n_images)
        ):
            if idx >= n_images:
                break
            image_size = self.image_size
            image = batch["image"]
            captions = batch["answer"]
            image = image.resize((image_size, image_size), Image.BILINEAR)
            image.save(self.real_dir / f"{idx}.png")

            prompts.append(captions[0])

        self.prompts = prompts
        with open(self.prompts_path, "wb") as f:
            pickle.dump(self.prompts, f)


def load_datasets(config: Config) -> dict[str, EvalDataset]:
    out = {}
    for name in config.evaluation.datasets:
        match name:
            case "cifar10":
                out[name] = Cifar10(config)
            case "coco_captions":
                out[name] = COCOCaptions(config)
            case _:
                raise ValueError("Unknown dataset: {name}")
    return out
