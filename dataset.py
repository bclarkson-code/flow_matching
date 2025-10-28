import torch
import io
import torchvision
from typing import Self
from PIL import Image
import h5py


class HDF5ImageTextDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading preprocessed images and text embeddings from HDF5"""

    def __init__(self, hdf5_path: str, normalize_images: bool = True):
        """
        Args:
            hdf5_path: Path to HDF5 file
            normalize_images: Whether to normalize images to [0, 1] range
        """

        self.hdf5_path = hdf5_path
        self.normalize_images = normalize_images
        self.start = 0
        self.num_repeats = 0

        with h5py.File(hdf5_path, "r") as f:
            self.total_samples: int = f.attrs["total_samples"]
            self.image_size = f.attrs["image_size"]
            self.seq_len = f.attrs["seq_len"]
            self.hidden_dim = f.attrs["hidden_dim"]

    def take(self, n: int) -> Self:
        self.total_samples = min(n, self.total_samples)
        return self

    def skip(self, n: int) -> Self:
        self.start_idx = n
        return self

    def repeat(self, n: int) -> Self:
        self.num_repeats = n
        return self

    def _to_index(self, idx: int) -> Self:
        idx += self.start_idx

        if idx >= self.total_samples:
            raise IndexError("dataset index out of range")


    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        with h5py.File(self.hdf5_path, "r") as f:
            png_bytes = f["images"][idx].tobytes()
            image = Image.open(io.BytesIO(png_bytes))

            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.PILToTensor(),
                ]
            )
            image = transform(image).float()

            if self.normalize_images:
                image = image / 255.0

            text_embeds = f["text_embeds"][idx]
            text_embeds = torch.from_numpy(
                text_embeds.copy()
            ).float()  

            attention_mask = f["attention_masks"][idx]
            attention_mask = torch.from_numpy(attention_mask.copy())

        return {
            "image": image,
            "text_embeds": text_embeds,
            "attention_mask": attention_mask,
        }
