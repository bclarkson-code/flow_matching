import numpy as np
import numpy.typing as npt
import torch
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3


def get_inception_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(
    real_features: npt.NDArray[np.float32], generated_features: npt.NDArray[np.float32]
) -> float:
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = (
        generated_features.mean(axis=0),
        np.cov(generated_features, rowvar=False),
    )

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):  # type: ignore
        covmean = covmean.real  # type: ignore

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)  # type: ignore
    return fid


def create_inception_model(device: torch.device) -> torch.nn.Module:
    inception_model = inception_v3(
        transform_input=False, weights=Inception_V3_Weights.DEFAULT
    )
    inception_model.fc = torch.nn.Identity()  # pyright: ignore[reportAttributeAccessIssue]
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model
