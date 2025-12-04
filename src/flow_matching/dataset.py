import io
from pathlib import Path

import grain.python as grain
import numpy as np
import torch
from array_record.python.array_record_data_source import ArrayRecordDataSource

from flow_matching.config import Config


class ToTorchTensors(grain.MapTransform):
    """Convert numpy/SharedMemoryArray to torch tensors."""

    def map(self, batch: dict) -> dict:
        return {
            "latents": torch.as_tensor(np.asarray(batch["latents"])).float(),
            "text_embeds": torch.as_tensor(np.asarray(batch["embeds"])).float(),
            "attention_mask": torch.as_tensor(np.asarray(batch["mask"])),
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
    resume_from_step: int | None = None,
) -> grain.DataLoader | grain.DatasetIterator:
    """Create Grain dataloader for training."""

    arrayrecord_paths = sorted(
        str(p)
        for p in Path(config.dataset.train_arrayrecord_path).glob("*.arrayrecord")
    )

    data_source = ArrayRecordDataSource(arrayrecord_paths)

    sampler = grain.IndexSampler(
        num_records=len(data_source),
        num_epochs=None,
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
            grain.Batch(batch_size=config.training.batch_size, drop_remainder=True),
            ToTorchTensors(),
        ],
        worker_count=config.dataset.num_workers,
        worker_buffer_size=config.dataset.prefetch_batches,
    )

    if resume_from_step is not None and resume_from_step > 0:
        loader_iter = iter(loader)
        state = loader_iter.get_state()
        state["last_seen_indices"]["sampler"] = (
            resume_from_step * config.training.batch_size
        )
        loader_iter.set_state(state)
        return loader_iter

    return loader


def create_eval_dataloader(
    config: Config,
    rank: int = 0,
    world_size: int = 1,
) -> grain.DataLoader:
    """Create Grain dataloader for evaluation."""

    arrayrecord_paths = sorted(
        str(p) for p in Path(config.dataset.eval_arrayrecord_path).glob("*.arrayrecord")
    )

    data_source = ArrayRecordDataSource(arrayrecord_paths)

    sampler = grain.IndexSampler(
        num_records=config.dataset.eval_samples
        if config.dataset.eval_samples is not None
        else len(data_source),
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
            grain.Batch(batch_size=config.training.batch_size, drop_remainder=False),
            ToTorchTensors(),
        ],
        worker_count=config.dataset.num_workers,
        worker_buffer_size=2,
    )

    return loader
