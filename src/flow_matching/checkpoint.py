import glob
import os
import re
from typing import Literal

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from flow_matching.config import Config
from flow_matching.dataset import create_eval_dataset, create_train_dataset
from flow_matching.distributed import is_main_process
from flow_matching.model import DiffusionTransformer, create_model_and_optimizer


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    config: Config,
) -> None:
    if not is_main_process():
        return

    if not os.path.exists(checkpoint_dir):
        return

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))

    if not checkpoint_files:
        return

    checkpoints = []
    for file in checkpoint_files:
        match = re.search(r"checkpoint_step_(\d+)\.pt", file)
        if match:
            checkpoints.append((int(match.group(1)), file))

    checkpoints.sort(reverse=True)

    keep_recent = config.checkpoint.keep_recent_checkpoints
    keep_every_n = config.checkpoint.keep_checkpoint_every_n_steps

    to_keep = set()

    for i, (step, path) in enumerate(checkpoints):
        if i < keep_recent:
            to_keep.add(path)
        elif step % keep_every_n == 0:
            to_keep.add(path)

    deleted_count = 0
    for step, path in checkpoints:
        if path not in to_keep:
            print(f"Removing old checkpoint: {path}")
            os.remove(path)
            deleted_count += 1

    if deleted_count > 0:
        print(f"Cleaned up {deleted_count} old checkpoint(s)")


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))

    if not checkpoint_files:
        return None

    step_numbers = []
    for file in checkpoint_files:
        match = re.search(r"checkpoint_step_(\d+)\.pt", file)
        if match:
            step_numbers.append((int(match.group(1)), file))

    if not step_numbers:
        return None

    latest_checkpoint = max(step_numbers, key=lambda x: x[0])[1]
    return latest_checkpoint


def save_checkpoint(
    model: DiffusionTransformer | DDP,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    step: int,
    loss: float,
    config: Config,
) -> None:
    if not is_main_process():
        return

    checkpoint_path = os.path.join(config.checkpoint.checkpoint_dir, f"checkpoint_step_{step}.pt")

    model_state_dict = (
        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": loss,
            "config": config,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    config: Config,
    device: torch.device,
) -> tuple[
    int,
    DDP | DiffusionTransformer,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.SequentialLR,
]:
    print(f"Loading checkpoint from {checkpoint_path}")
    model, optimiser, scheduler = create_model_and_optimizer(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimiser.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    step = checkpoint["step"]
    print(f"Resumed from step {step}")

    return step, model, optimiser, scheduler


def resume_from_checkpoint(
    resume: str | Literal["latest"],
    config: Config,
    device: torch.device,
) -> tuple[
    DiffusionTransformer | DDP,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.SequentialLR,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    if resume == "latest":
        checkpoint_path = find_latest_checkpoint(config.checkpoint.checkpoint_dir)
    else:
        checkpoint_path = resume
    print(f"Resuming from latest checkpoint: {checkpoint_path}")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint: {checkpoint_path}")

    step, model, optimiser, scheduler = load_checkpoint(
        checkpoint_path=checkpoint_path, device=device, config=config
    )
    train_dataset = create_train_dataset(config=config, resume_from_step=step)
    eval_dataset = create_eval_dataset(config=config)
    return model, optimiser, scheduler, train_dataset, eval_dataset
