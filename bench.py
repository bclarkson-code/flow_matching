import argparse
import math
import os
import time
from dataclasses import replace

import hydra
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from torch.profiler import ProfilerActivity

from flow_matching.config import Config
from train import train_worker


GIGABYTE = 1024**3


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark and debug transformer")
    parser.add_argument(
        "--method",
        type=str,
        default="default",
    )
    return parser.parse_args()


def profile(
    config: Config,
) -> None:
    config = replace(
        config,
        use_wandb=False,
        eval_samples=None,
        num_steps=25,
        train_dataset_pattern="data/text-to-image-2M_64x64_preprocessed-{000001..00037}.tar",
    )
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        train_worker(0, 1, config)
    prof.export_chrome_trace("pytorch_trace.json")


def try_batch_size(
    batch_size: int,
    device: torch.device,
    config: Config,
    overhead: float = 0.9,
) -> tuple[bool, float | None, float | None]:
    os.environ["TQDM_DISABLE"] = "1"

    config = replace(
        config,
        distributed=False,
        use_wandb=False,
        eval_samples=None,
        batch_size=batch_size,
        num_inference_steps=50,
        num_steps=5,
        train_dataset_pattern="data/text-to-image-2M_64x64_preprocessed-{000001..00037}.tar",
    )
    try:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        train_worker(0, 1, config)
    except torch.OutOfMemoryError:
        return False, None, None
    finally:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    peak_mem = torch.cuda.max_memory_allocated() / GIGABYTE
    print(f"{ peak_mem=}")
    total_mem = torch.cuda.get_device_properties(device).total_memory / GIGABYTE

    usage_frac = peak_mem / total_mem

    os.environ["TQDM_DISABLE"] = ""

    return usage_frac < overhead, usage_frac, peak_mem


def find_batch_size(config: Config) -> int | None:
    batch_size = 2

    best_fit = None

    while True:
        fits, usage, gbs = try_batch_size(
            batch_size=batch_size, device=torch.device("cuda:0"), config=config
        )
        batch_size = batch_size * 2
        if fits:
            print(f"FITS: {batch_size=}, {usage=} ({gbs:.2f} GB)")
            best_fit = batch_size
        else:
            print(f"DOES NOT FIT: {batch_size=}")
            break
    return best_fit


@hydra.main(version_base=None, config_path="configs", config_name="config")
def benchmark(cfg: DictConfig):
    """
    Time how long it takes to pass 2**16 images through the model, with evaluation on
    2**10 images at the start and end
    """
    config: Config = OmegaConf.to_object(cfg)  # type: ignore

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    images_per_step = (
        config.training.batch_size
        * config.training.gradient_accumulation_steps
        * world_size
    )
    num_steps = math.ceil(2**16 / images_per_step)
    print(f"Processing {images_per_step} images per step")
    print(f"Running for {num_steps} steps")

    config.logging.use_wandb = False
    config.training.num_steps = num_steps

    start = time.time()
    if config.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        train_worker(local_rank, world_size, config)
    else:
        train_worker(0, 1, config)
    duration = time.time() - start

    print(f"Speedrun took: {duration:.3f} seconds")


if __name__ == "__main__":
    benchmark()
