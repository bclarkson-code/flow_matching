import logging
import math
import random
from functools import partial

import torch
import webdataset as wds
from webdataset import WebDataset  # type: ignore

from flow_matching.config import Config
from flow_matching.distributed import is_main_process


logger = logging.getLogger()


def create_train_dataset(
    config: Config,
    resume_from_step: None | int = None,
) -> WebDataset:
    train_dataset = WebDataset(
        config.train_dataset_pattern,
        shardshuffle=1000,
        nodesplitter=wds.split_by_node if config.distributed else None,
    )

    train_dataset = train_dataset.shuffle(
        config.shuffle_buffer_size, rng=random.Random(config.seed)
    )

    train_dataset = (
        train_dataset.decode("pilrgb")
        .map(
            lambda row: {
                "latents": torch.from_numpy(row["latents.pyd"]).float().squeeze(),
                "text_embeds": torch.from_numpy(row["embeds.pyd"]).float().squeeze(),
                "attention_mask": torch.from_numpy(row["mask.pyd"]).squeeze(),
            }
        )
        .repeat()
        .batched(config.batch_size)
    )

    if config.dataset_size is not None:
        train_dataset = train_dataset.slice(0, config.dataset_size)

    if resume_from_step is not None:
        train_dataset = train_dataset.slice(resume_from_step)

    if is_main_process():
        logger.info(f"Loaded WebDataset from {config.train_dataset_pattern}")
    return train_dataset


def create_eval_dataset(
    config: Config,
) -> WebDataset:
    eval_dataset = WebDataset(
        config.eval_dataset_pattern,
        shardshuffle=False,
        nodesplitter=wds.split_by_node if config.distributed else None,
    )

    eval_dataset = (
        eval_dataset.decode("pilrgb")
        .map(
            lambda row: {
                "latents": torch.from_numpy(row["latents.pyd"]).float().squeeze(),
                "text_embeds": torch.from_numpy(row["embeds.pyd"]).float().squeeze(),
                "attention_mask": torch.from_numpy(row["mask.pyd"]).squeeze(),
            }
        )
        .batched(config.batch_size)
    )

    if config.eval_samples is not None:
        n_batches = math.ceil(config.eval_samples / config.batch_size)
        eval_dataset = eval_dataset.slice(0, n_batches, 1)

    if is_main_process():
        logger.info(f"Loaded WebDataset from {config.eval_dataset_pattern}")
    return eval_dataset
