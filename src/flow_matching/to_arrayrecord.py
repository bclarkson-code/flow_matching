import io
import multiprocessing as mp
from pathlib import Path

import numpy as np
import webdataset as wds
from array_record.python.array_record_module import ArrayRecordWriter
from tqdm import tqdm


def convert_single_shard(args: tuple[str, str, int]) -> tuple[str, int]:
    """Convert one webdataset shard to one ArrayRecord file."""
    input_tar, output_path, i = args

    dataset = wds.WebDataset(input_tar, shardshuffle=False).decode()
    writer = ArrayRecordWriter(output_path)

    count = 0
    for sample in tqdm(dataset, leave=False, position=i + 1, desc=str(i), total=10_000):
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            latents=sample["latents.pyd"],
            embeds=sample["embeds.pyd"],
            mask=sample["mask.pyd"],
        )
        writer.write(buffer.getvalue())
        count += 1

    writer.close()
    return Path(input_tar).stem, count


def convert_webdataset_parallel(
    input_pattern: str,
    output_dir: str,
    num_workers: int = 16,
    desc: str = "Converting",
):
    """Convert webdataset shards matching pattern in parallel."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand the brace pattern to get individual tar paths
    tar_files = sorted(wds.shardlists.expand_urls(input_pattern))

    conversion_args = [
        (tar_path, str(output_dir / f"{Path(tar_path).stem}.arrayrecord"), i)
        for i, tar_path in enumerate(tar_files)
    ]

    total_samples = 0

    with mp.Pool(num_workers) as pool:
        with tqdm(
            total=len(conversion_args), desc=f"{desc} (shards)", position=0
        ) as shard_pbar:
            for shard_name, count in pool.imap_unordered(
                convert_single_shard, conversion_args
            ):
                total_samples += count
                shard_pbar.update(1)
                shard_pbar.set_postfix({"samples": total_samples, "last": shard_name})

    print(
        f"Done. Converted {total_samples:,} samples across {len(tar_files)} files to {output_dir}"
    )


if __name__ == "__main__":
    base_dir = "/mnt/storage/datasets/flow_matching"

    # Train: shards 000001-000230
    print("Converting train set...")
    convert_webdataset_parallel(
        input_pattern=f"{base_dir}/text-to-image-2M_64x64_preprocessed-{{000001..000230}}.tar",
        output_dir=f"{base_dir}/jackyhate/train",
        num_workers=31,
        desc="Train",
    )

    # Eval: shard 000000 only
    print("\nConverting eval set...")
    convert_webdataset_parallel(
        input_pattern=f"{base_dir}/text-to-image-2M_64x64_preprocessed-000000.tar",
        output_dir=f"{base_dir}/jackyhate/eval",
        num_workers=1,
        desc="Eval",
    )
