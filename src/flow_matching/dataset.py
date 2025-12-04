import io

import grain.python as grain
import numpy as np
import torch
from array_record.python.array_record_data_source import ArrayRecordDataSource

from flow_matching.config import Config


class ToTorchTensors(grain.MapTransform):
    """Convert numpy arrays to torch tensors."""

    def map(self, sample: dict) -> dict:
        return {
            "latents": torch.from_numpy(sample["latents"]).float(),
            "text_embeds": torch.from_numpy(sample["embeds"]).float(),
            "attention_mask": torch.from_numpy(sample["mask"]),
        }


class DeserializeSample(grain.MapTransform):
    """Deserialize npz bytes back to numpy arrays."""

    def map(self, raw_bytes: bytes) -> dict:
        buffer = io.BytesIO(raw_bytes)
        data = np.load(buffer)
        return {
            "latents": data["latents"],
            "embeds": data["embeds"],
            "mask": data["mask"],
        }


def create_train_dataloader(
    config: Config,
    rank: int = 0,
    world_size: int = 1,
) -> grain.DataLoader:
    """Create Grain dataloader for training."""

    data_source = ArrayRecordDataSource(config.dataset.train_arrayrecord_path)

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None,  # Infinite iteration
        shard_options=grain.ShardOptions(
            shard_index=rank,
            shard_count=world_size,
            drop_remainder=True,
        ),
        shuffle=True,
        seed=config.seed,
    )

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            DeserializeSample(),
            ToTorchTensors(),
            grain.Batch(batch_size=config.training.batch_size, drop_remainder=True),
        ],
        worker_count=config.dataset.num_workers,
        worker_buffer_size=config.dataset.prefetch_batches,
    )

    return loader


def create_eval_dataloader(
    config: Config,
    rank: int = 0,
    world_size: int = 1,
) -> grain.DataLoader:
    """Create Grain dataloader for evaluation."""

    data_source = ArrayRecordDataSource(config.dataset.eval_arrayrecord_path)

    # For eval: no shuffle, single epoch
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=rank,
            shard_count=world_size,
            drop_remainder=False,
        ),
        shuffle=False,
        seed=config.seed,
    )

    loader = grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[
            DeserializeSample(),
            ToTorchTensors(),
            grain.Batch(batch_size=config.training.batch_size, drop_remainder=False),
        ],
        worker_count=config.dataset.num_workers,
        worker_buffer_size=2,
    )

    return loader
