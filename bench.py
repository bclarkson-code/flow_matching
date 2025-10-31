import math
import os
from dataclasses import dataclass, replace

import torch
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity

from flow_matching.config import Config, ConfigType
from train import train_worker


GIGABYTE = 1024**3


@dataclass
class Args:
    resume: bool = False


def profile(
    config: Config,
) -> None:
    config = replace(
        config,
        distributed=True,
        use_wandb=False,
        eval_samples=None,
        gradient_accumulation_steps=2,
        batch_size=128,
        num_inference_steps=50,
        num_steps=5,
    )
    args = Args()
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        train_worker(0, 1, config, args)
    prof.export_chrome_trace("pytorch_trace.json")


def try_batch_size(
    batch_size: int,
    device: torch.device,
    config: Config,
    overhead: float = 0.9,
) -> tuple[bool, float | None, float | None]:
    os.environ["TQDM_DISABLE"] = "1"
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, "w")

    config = replace(
        config,
        distributed=False,
        use_wandb=False,
        eval_samples=None,
        batch_size=batch_size,
        num_inference_steps=50,
        num_steps=5,
    )
    args = Args()
    try:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        train_worker(0, 1, config, args)
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


def benchmark(config: Config) -> float:
    """
    Time how long it takes to pass 2**16 images through the model, with evaluation on
    2**10 images at the start and end
    """
    images_per_step = (
        config.batch_size * config.gradient_accumulation_steps * config.world_size
    )
    num_steps = math.ceil(2**16 / images_per_step)
    print(f"Processing {images_per_step} images per step")
    print(f"Running for {num_steps} steps")

    config = replace(
        config,
        use_wandb=False,
        num_steps=num_steps,
        eval_every=num_steps,
        eval_samples=1024,
    )

    args = Args()

    if config.distributed:
        world_size = config.world_size
        durations = mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True,
        )
        print(durations)
        duration = max(durations)
    else:
        duration = train_worker(0, 1, config, args)

    print(f"Speedrun took: {duration:.3f} seconds")
    return duration


if __name__ == "__main__":
    config = ConfigType.FULL_SCALE.to_config()
    profile(config)
    # batch_size = find_batch_size(config)
    # batch_size = 128
    # print(f"Found batch size: {batch_size}")
    # config = replace(config, batch_size=batch_size)
    # duration = benchmark(config)
