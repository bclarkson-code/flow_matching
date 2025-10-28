from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets
from dataset import HDF5ImageTextDataset
import torch
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
from config import Config, FullScaleConfig
from transformers import T5EncoderModel, T5Tokenizer


class TextEmbedder(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(
            config.model.text_embed_model_string
        )
        self.text_encoder.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.model.text_embed_model_string, legacy=True
        )
        self.max_length = config.model.text_max_length

    def forward(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            tokens = {k: v.to(self.text_encoder.device) for k, v in tokens.items()}
            text_embeddings = self.text_encoder(tokens["input_ids"]).last_hidden_state
            return text_embeddings, tokens["attention_mask"]


def preprocess_and_resize(row: dict, target_size: int = 64) -> dict:
    """Resize image to target size and keep metadata"""
    image = row["jpg"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((target_size, target_size), Image.LANCZOS)

    return {"jpg": image, "json": row["json"]}


def process_and_save_to_hdf5(
    dataset_path: str,
    output_hdf5_path: str,
    config: Config,
    batch_size: int = 32,
    use_fp16: bool = True,
):
    """
    Load dataset from disk, process images and text embeddings, and save to HDF5.

    Args:
        dataset_path: Path to the saved HuggingFace dataset
        output_hdf5_path: Path for output HDF5 file
        config: Config object
        batch_size: Batch size for text encoding
        use_fp16: Whether to store embeddings as float16 (saves 50% space)
    """
    import io

    print("Loading dataset from disk...")
    dataset = load_from_disk(dataset_path)
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    # Initialize text embedder
    print("Loading text encoder...")
    text_embedder = TextEmbedder(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedder = text_embedder.to(device)
    print(f"Text encoder loaded on {device}")

    # Get dimensions from first sample
    first_sample = dataset[0]
    image = first_sample["jpg"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    image_size = image.size[0]  # Assuming square images
    text = first_sample["json"]["prompt"]

    # Get embedding dimensions
    with torch.no_grad():
        sample_embeds, sample_mask = text_embedder([text])
        embed_shape = sample_embeds.shape  # [1, seq_len, hidden_dim]
        seq_len = embed_shape[1]
        hidden_dim = embed_shape[2]

    print(f"Image size: {image_size}x{image_size}")
    print(f"Text embedding shape: [{seq_len}, {hidden_dim}]")

    # Create HDF5 file with datasets
    print(f"Creating HDF5 file: {output_hdf5_path}")
    dtype_embeds = np.float16 if use_fp16 else np.float32

    with h5py.File(output_hdf5_path, "w") as f:
        # Use variable-length dtype for PNG bytes
        dt = h5py.special_dtype(vlen=np.uint8)

        # Create datasets
        images_dset = f.create_dataset(
            "images",
            shape=(total_samples,),
            dtype=dt,  # Variable-length bytes
        )

        text_embeds_dset = f.create_dataset(
            "text_embeds",
            shape=(total_samples, seq_len, hidden_dim),
            dtype=dtype_embeds,
            compression="gzip",
            compression_opts=4,
            chunks=(batch_size, seq_len, hidden_dim),
        )

        attention_masks_dset = f.create_dataset(
            "attention_masks",
            shape=(total_samples, seq_len),
            dtype=np.bool_,
            compression="gzip",
            compression_opts=4,
            chunks=(batch_size, seq_len),
        )

        # Store metadata
        f.attrs["total_samples"] = total_samples
        f.attrs["image_size"] = image_size
        f.attrs["seq_len"] = seq_len
        f.attrs["hidden_dim"] = hidden_dim
        f.attrs["use_fp16"] = use_fp16
        f.attrs["image_format"] = "PNG"

        # Process in batches
        print("Processing and saving data...")

        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            batch = dataset[start_idx:end_idx]

            # Process images and texts
            png_bytes_batch = []
            text = [t['prompt'] for t in batch['json']]

            for image in batch['jpg']:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Convert PIL image to PNG bytes
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                png_bytes = buffer.getvalue()
                png_bytes_batch.append(np.frombuffer(png_bytes, dtype=np.uint8))

            # Encode texts
            with torch.no_grad():
                text_embeds, attention_masks = text_embedder(text)
                text_embeds = text_embeds.cpu().numpy().astype(dtype_embeds)
                attention_masks = attention_masks.cpu().numpy().astype(np.bool_)

            # Save to HDF5
            for i, (png_bytes, idx) in enumerate(
                zip(png_bytes_batch, range(start_idx, end_idx))
            ):
                images_dset[idx] = png_bytes

            text_embeds_dset[start_idx:end_idx] = text_embeds
            attention_masks_dset[start_idx:end_idx] = attention_masks

    print(f"Done! Saved {total_samples} samples to {output_hdf5_path}")

    # Print file size
    import os

    file_size_gb = os.path.getsize(output_hdf5_path) / (1024**3)
    print(f"File size: {file_size_gb:.2f} GB")


def download_and_save_in_chunks(
    dataset_name: str,
    output_path: str,
    target_size: int = 64,
    split: str = "train",
    chunk_size: int = 10_000,
):
    """Download and save in chunks to avoid memory issues"""
    print(f"Loading {dataset_name} in streaming mode...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    chunk_idx = 0
    samples = []

    for idx, sample in enumerate(tqdm(dataset)):
        resized = preprocess_and_resize(sample, target_size)
        samples.append(resized)

        if (idx + 1) % chunk_size == 0:
            chunk_dataset = Dataset.from_list(samples)
            chunk_path = f"{output_path}_chunk_{chunk_idx}"
            chunk_dataset.save_to_disk(chunk_path)
            print(f"Saved chunk {chunk_idx} ({len(samples)} samples)")

            chunk_idx += 1
            samples = []

    if samples:
        chunk_dataset = Dataset.from_list(samples)
        chunk_path = f"{output_path}_chunk_{chunk_idx}"
        chunk_dataset.save_to_disk(chunk_path)
        print(f"Saved final chunk {chunk_idx} ({len(samples)} samples)")

    print("Concatenating chunks...")

    chunks = [load_from_disk(f"{output_path}_chunk_{i}") for i in range(chunk_idx + 1)]
    final_dataset = concatenate_datasets(chunks)
    final_dataset.save_to_disk(output_path)

    print(f"Done! Saved {len(final_dataset)} samples to {output_path}")


# Run this once
if __name__ == "__main__":
    config = FullScaleConfig()

    # Convert the HuggingFace dataset to HDF5 with text embeddings
    process_and_save_to_hdf5(
        dataset_path=config.dataset_path,
        output_hdf5_path="data/text-to-image-2M_64x64_preprocessed.h5",
        config=config,
        batch_size=512,  # Adjust based on your GPU memory
        use_fp16=True,  # Use float16 for embeddings to save space
    )

    # Test loading the dataset
    print("\nTesting dataset loading...")
    test_dataset = HDF5ImageTextDataset(
        hdf5_path="data/text-to-image-2M_64x64_preprocessed.h5", normalize_images=True
    )

    print(f"Dataset length: {len(test_dataset)}")
    sample = test_dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
    print(
        f"Text embeds shape: {sample['text_embeds'].shape}, dtype: {sample['text_embeds'].dtype}"
    )
    print(
        f"Attention mask shape: {sample['attention_mask'].shape}, dtype: {sample['attention_mask'].dtype}"
    )

    # Test with DataLoader
    print("\nTesting DataLoader...")
    train_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=True, num_workers=2
    )

    batch = next(iter(train_loader))
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch text embeds shape: {batch['text_embeds'].shape}")
    print(f"Batch attention mask shape: {batch['attention_mask'].shape}")
