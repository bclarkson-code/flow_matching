import logging
import random

import torch
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset import WebDataset  # type: ignore

from flow_matching.config import Config
from flow_matching.distributed import is_main_process


logger = logging.getLogger()


def create_train_dataset(
    config: Config,
    resume_from_step: None | int = None,
) -> DataLoader:
    train_dataset = WebDataset(
        config.dataset.train_dataset_pattern,
        shardshuffle=config.dataset.shuffle_buffer_size,
        nodesplitter=wds.split_by_node if config.distributed.distributed else None,  # type: ignore
    )

    train_dataset = (
        train_dataset.decode("pilrgb")
        .shuffle(config.dataset.shuffle_buffer_size, rng=random.Random(config.seed))
        .map(
            lambda row: {
                "latents": torch.from_numpy(row["latents.pyd"]).float(),
                "text_embeds": torch.from_numpy(row["embeds.pyd"]).float(),
                "attention_mask": torch.from_numpy(row["mask.pyd"]),
            }
        )
    )

    if config.dataset.dataset_size is not None:
        train_dataset = train_dataset.slice(0, config.dataset.dataset_size)
    if resume_from_step is not None:
        train_dataset = train_dataset.slice(resume_from_step)

    train_dataset = train_dataset.batched(config.training.batch_size).repeat()

    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=config.dataset.num_workers,
        prefetch_factor=config.dataset.prefetch_batches,
        persistent_workers=True,
        pin_memory=True,
    )

    if is_main_process():
        logger.info(
            f"Loaded WebDataset with {config.dataset.num_workers} workers from {config.dataset.train_dataset_pattern}"
        )

    return train_loader


def create_eval_dataset(
    config: Config,
) -> WebDataset:
    eval_dataset = WebDataset(
        config.dataset.eval_dataset_pattern,
        shardshuffle=False,
        nodesplitter=wds.split_by_node if config.distributed.distributed else None,  # type: ignore
    )

    eval_dataset = eval_dataset.decode("pilrgb").map(
        lambda row: {
            "latents": torch.from_numpy(row["latents.pyd"]).float().squeeze(),
            "text_embeds": torch.from_numpy(row["embeds.pyd"]).float().squeeze(),
            "attention_mask": torch.from_numpy(row["mask.pyd"]).squeeze(),
        }
    )

    if config.dataset.eval_samples is not None:
        eval_dataset = eval_dataset.slice(config.dataset.eval_samples)

    eval_dataset = eval_dataset.batched(config.training.batch_size)

    if is_main_process():
        logger.info(f"Loaded WebDataset from {config.dataset.eval_dataset_pattern}")
    return eval_dataset
