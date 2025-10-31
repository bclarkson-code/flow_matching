import logging
import os

import torch
import torch.distributed as dist

from flow_matching.config import Config


def setup_distributed(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    torch.cuda.set_device(rank)

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        force=True, 
    )


def cleanup_distributed() -> None:
    dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0
