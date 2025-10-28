import torch
from dataset import HDF5ImageTextDataset
from dataclasses import asdict
from pprint import pprint
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import numpy as np
import os
from pathlib import Path
import time
from model import DiffusionTransformer
from datasets import load_from_disk
import torchvision
from PIL import Image
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from tqdm import tqdm
from config import Config, ConfigType
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import argparse
import glob
import re


def setup_distributed(rank: int, world_size: int, config: Config) -> None:
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = config.master_addr
    os.environ["MASTER_PORT"] = config.master_port
    dist.init_process_group(config.backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Get the rank of the current process."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_inception_features(images: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    with torch.no_grad():
        images = torch.nn.functional.interpolate(
            images, size=(299, 299), mode="bilinear", align_corners=False
        )
        features = model(images)
    return features.cpu().numpy()


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = generated_features.mean(axis=0), np.cov(
        generated_features, rowvar=False
    )

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def generate_images(
    model: DiffusionTransformer,
    latents: torch.Tensor,
    text: list[str],
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        batch_size = latents.shape[0]
        generated_latents = torch.randn_like(latents)

        for t_idx in range(config.num_inference_steps):
            t = torch.ones(batch_size, device=device) * (t_idx / config.num_inference_steps)
            pred_v = model(image_latents=generated_latents, text=text, time=t)
            dt = 1.0 / config.num_inference_steps
            generated_latents = generated_latents + pred_v * dt

        generated_images = model.image_embedder.from_latent(generated_latents)
        return torch.clamp(generated_images, 0, 1)


def transform_image(
    batch: dict[str, list[Image.Image | dict[str, str]]],
) -> dict[str, torch.Tensor | str]:
    images = []
    texts = []
    for item in batch:
        image = item["jpg"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                torchvision.transforms.ConvertImageDtype(torch.float),
            ]
        )
        images.append(transform(image))
        texts.append(item["json"]["prompt"])

    return {"image": torch.stack(images), "text": texts}


def compute_eval_loss(
    model: DiffusionTransformer,
    image: torch.Tensor,
    text: list[str],
    device: torch.device,
) -> float:
    latents = model.image_embedder.to_latent(image)
    batch_size = latents.shape[0]

    noise = torch.randn_like(latents).to(device)
    time = torch.rand(batch_size).to(device)
    target_velocity = latents - noise
    noisy_latents = noise + time.view(-1, 1, 1, 1) * (latents - noise)
    predicted_velocity = model(image_latents=noisy_latents, text=text, time=time)
    loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    return loss.item()


@torch.no_grad()
def evaluate(
    model: DiffusionTransformer | DDP,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    inception_model: torch.nn.Module,
    step: int,
    config: Config,
) -> tuple[float, float]:
    if not is_main_process():
        return 0.0, 0.0

    model.eval()
    with torch.no_grad():
        unwrapped_model = model.module if isinstance(model, DDP) else model

        eval_losses = []
        real_features_list = []
        generated_features_list = []
        generated_images_list = []
        text_input = []

        for batch in tqdm(eval_dataloader, desc="Evaluating", position=1, leave=False):
            text = batch["text"]
            text_input.extend(text)
            image = batch["image"]
            image = (image * 2) - 1
            image = image.to(device)

            latents = unwrapped_model.image_embedder.to_latent(image)

            loss = compute_eval_loss(unwrapped_model, image, text, device)
            eval_losses.append(loss)

            generated_images = generate_images(
                unwrapped_model, latents, text, config, device
            )

            image_normalized = (image + 1) / 2
            real_features = get_inception_features(image_normalized, inception_model)
            generated_features = get_inception_features(generated_images, inception_model)

            real_features_list.append(real_features)
            generated_features_list.append(generated_features)
            generated_images_list.append(generated_images.cpu().numpy())

        avg_eval_loss = float(np.mean(eval_losses))

        all_real_features = np.concatenate(real_features_list, axis=0)
        all_generated_features = np.concatenate(generated_features_list, axis=0)
        fid_score = calculate_fid(all_real_features, all_generated_features)

    if config.use_wandb:
        images_for_wandb = [
            wandb.Image(img) for img in np.concatenate(generated_images_list)
        ]

        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/fid": fid_score,
                "eval/generated_images": images_for_wandb,
                "eval/text_input": text_input,
            },
            step=step,
        )

    model.train()
    return avg_eval_loss, fid_score


def log_training_metrics(
    loss: float,
    grad_norm: torch.Tensor,
    optimiser: torch.optim.AdamW,
    latents: torch.Tensor,
    noise: torch.Tensor,
    time: torch.Tensor,
    noisy_latents: torch.Tensor,
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    step: int,
    config: Config,
) -> None:
    if not is_main_process() or not config.use_wandb:
        return

    wandb.log(
        {
            "train/loss": loss,
            "train/grad_norm": grad_norm.item(),
            "train/learning_rate": optimiser.param_groups[0]["lr"],
            "train/latents_min": latents.min().item(),
            "train/latents_mean": latents.mean().item(),
            "train/latents_max": latents.max().item(),
            "train/latents_std": latents.std().item(),
            "train/noise_min": noise.min().item(),
            "train/noise_mean": noise.mean().item(),
            "train/noise_max": noise.max().item(),
            "train/noise_std": noise.std().item(),
            "train/time_min": time.min().item(),
            "train/time_max": time.max().item(),
            "train/time_std": time.std().item(),
            "train/noisy_latents_min": noisy_latents.min().item(),
            "train/noisy_latents_mean": noisy_latents.mean().item(),
            "train/noisy_latents_max": noisy_latents.max().item(),
            "train/noisy_latents_std": noisy_latents.std().item(),
            "train/predicted_velocity_min": predicted_velocity.min().item(),
            "train/predicted_velocity_mean": predicted_velocity.mean().item(),
            "train/predicted_velocity_max": predicted_velocity.max().item(),
            "train/predicted_velocity_std": predicted_velocity.std().item(),
            "train/target_velocity_min": target_velocity.min().item(),
            "train/target_velocity_mean": target_velocity.mean().item(),
            "train/target_velocity_max": target_velocity.max().item(),
            "train/target_velocity_std": target_velocity.std().item(),
        },
        step=step,
    )


def train_step(
    model: DiffusionTransformer | DDP,
    dataloader_iter: iter,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    step: int,
    config: Config,
) -> tuple[DiffusionTransformer | DDP, float]:
    optimiser.zero_grad()

    unwrapped_model = model.module if isinstance(model, DDP) else model

    total_loss = 0.0
    all_latents = []
    all_noise = []
    all_time = []
    all_noisy_latents = []
    all_predicted_velocity = []
    all_target_velocity = []

    for _ in range(config.gradient_accumulation_steps):
        batch = next(dataloader_iter)

        text = batch["text"]
        image = batch["image"]
        image = (image * 2) - 1
        image = image.to(device)

        latents = unwrapped_model.image_embedder.to_latent(image)
        batch_size = latents.shape[0]

        noise = torch.randn_like(latents).to(device)
        time = torch.rand(batch_size).to(device)

        target_velocity = latents - noise
        noisy_latents = noise + time.view(-1, 1, 1, 1) * (latents - noise)
        predicted_velocity = model(image_latents=noisy_latents, text=text, time=time)
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)

        loss = loss / config.gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item()

        latents = latents.detach().cpu()
        noise = noise.detach().cpu()
        time = time.detach().cpu()
        noisy_latents = noisy_latents.detach().cpu()
        predicted_velocity = predicted_velocity.detach().cpu()
        target_velocity = target_velocity.detach().cpu()

        all_latents.append(latents)
        all_noise.append(noise)
        all_time.append(time)
        all_noisy_latents.append(noisy_latents)
        all_predicted_velocity.append(predicted_velocity)
        all_target_velocity.append(target_velocity)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=config.gradient_clip_max_norm
    )

    optimiser.step()
    scheduler.step()

    log_training_metrics(
        total_loss,
        grad_norm,
        optimiser,
        torch.cat(all_latents, dim=0),
        torch.cat(all_noise, dim=0),
        torch.cat(all_time, dim=0),
        torch.cat(all_noisy_latents, dim=0),
        torch.cat(all_predicted_velocity, dim=0),
        torch.cat(all_target_velocity, dim=0),
        step,
        config=config,
    )

    return model, total_loss


def save_checkpoint(
    model: DiffusionTransformer | DDP,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader_iter: iter,
    step: int,
    loss: float,
    config: Config,
) -> None:
    if not is_main_process():
        return

    checkpoint_path = os.path.join(config.checkpoint_dir, f"checkpoint_step_{step}.pt")

    dataset_state = {}
    if hasattr(dataloader_iter, "_dataset") and hasattr(
        dataloader_iter._dataset, "state_dict"
    ):
        dataset_state = dataloader_iter._dataset.state_dict()

    model_state_dict = (
        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    )

    torch.save(
        {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimiser.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "dataset_state_dict": dataset_state,
            "loss": loss,
            "config": config,
        },
        checkpoint_path,
    )


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

    keep_recent = config.keep_recent_checkpoints
    keep_every_n = config.keep_checkpoint_every_n_steps

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


def load_checkpoint(
    checkpoint_path: str,
    model: DiffusionTransformer | DDP,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> int:
    print(f"Loading checkpoint from {checkpoint_path}")
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

    return step


def create_train_dataset(
    config: Config,
    epoch: int = 0,
    dataset_state_dict: dict | None = None,
):
    if config.use_text_embedder:
        train_dataset = (
            load_from_disk(config.dataset_path)
            .skip(config.eval_samples)
            .shuffle(seed=config.seed + epoch)
        )
    else:
        train_dataset = HDF5ImageTextDataset(
            hdf5_path="data/text-to-image-2M_64x64_preprocessed.h5",
            normalize_images=True,
        )
    if config.dataset_size is not None:
        train_dataset = train_dataset.take(config.dataset_size).repeat(
            config.num_repeats
        )

    if config.distributed:
        rank = get_rank()
        world_size = config.world_size
        train_dataset = train_dataset.shard(num_shards=world_size, index=rank)

    if dataset_state_dict:
        print("Loading dataset state from checkpoint")
        train_dataset.load_state_dict(dataset_state_dict)
    print(f"Loaded: {len(train_dataset)} images")

    return train_dataset


def create_dataloaders(
    config: Config,
    epoch: int = 0,
    dataset_state_dict: dict | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_dataset = create_train_dataset(config, epoch, dataset_state_dict)

    eval_dataset = load_from_disk(config.dataset_path).take(config.eval_samples)

    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=transform_image,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=transform_image,
        pin_memory=True,
        persistent_workers=True if config.num_workers > 0 else False,
    )

    return dataloader, eval_dataloader


def create_model_and_optimizer(
    config: Config, device: torch.device
) -> tuple[DiffusionTransformer | DDP, torch.optim.AdamW, SequentialLR]:
    model = DiffusionTransformer(config)
    model = model.to(device)

    print(config)

    if config.distributed:
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    warmup_steps = int(config.warmup_ratio * config.num_steps)

    warmup_scheduler = LinearLR(
        optimiser,
        start_factor=config.lr_decay_ratio,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimiser,
        T_max=config.num_steps - warmup_steps,
        eta_min=config.learning_rate * config.lr_decay_ratio,
    )

    scheduler = SequentialLR(
        optimiser,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    return model, optimiser, scheduler


def create_inception_model(device: torch.device) -> torch.nn.Module:
    inception_model = inception_v3(
        transform_input=False, weights=Inception_V3_Weights.DEFAULT
    )
    inception_model.fc = torch.nn.Identity()
    inception_model = inception_model.to(device)
    inception_model.eval()
    return inception_model


def handle_step_logging_and_checkpointing(
    step: int,
    loss: float,
    model: DiffusionTransformer | DDP,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    inception_model: torch.nn.Module,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader_iter: iter,
    config: Config,
) -> None:
    if step % config.log_every == 0 and is_main_process():
        print(f"Step {step}, Loss: {loss:.4f}")

    if step % config.eval_every == 0 and config.eval_samples > 0:
        eval_loss, fid = evaluate(
            model, eval_dataloader, device, inception_model, step, config
        )
        eval_loss = fid = 1.0
        if is_main_process():
            print(f"Step {step}, Eval Loss: {eval_loss:.4f}, FID: {fid:.2f}")

    should_save = (
        config.save_checkpoints and step % config.checkpoint_freq == 0 and step > 0
    )
    if should_save:
        cleanup_old_checkpoints(config.checkpoint_dir, config)
        save_checkpoint(
            model, optimiser, scheduler, dataloader_iter, step, loss, config
        )


def training_loop(
    model: DiffusionTransformer | DDP,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    inception_model: torch.nn.Module,
    config: Config,
    start_step: int = 0,
) -> None:
    step = start_step
    epoch = 0

    pbar_context = tqdm(
        initial=start_step,
        total=config.num_steps,
        desc="Training",
        position=0,
        disable=not is_main_process(),
    )
    dataloader_iter = iter(dataloader)

    with pbar_context as pbar:
        while step < config.num_steps:
            try:
                model, loss = train_step(
                    model,
                    dataloader_iter,
                    optimiser,
                    scheduler,
                    device,
                    step,
                    config,
                )
            except StopIteration:
                epoch += 1
                dataloader, eval_dataloader = create_dataloaders(config, epoch=epoch)
                dataloader_iter = iter(dataloader)
                continue

            handle_step_logging_and_checkpointing(
                step,
                loss,
                model,
                eval_dataloader,
                device,
                inception_model,
                optimiser,
                scheduler,
                dataloader_iter,
                config,
            )

            step += 1
            pbar.update(1)

    eval_loss, fid = evaluate(
        model, eval_dataloader, device, inception_model, step, config
    )
    if is_main_process():
        print(f"Final Eval Loss: {eval_loss:.4f}, FID: {fid:.2f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion transformer")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from, or 'latest' to resume from the latest checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
    )
    return parser.parse_args()


def train_worker(rank: int, world_size: int, config: Config, args) -> None:
    """Worker function for distributed training."""
    if config.distributed:
        setup_distributed(rank, world_size, config)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(config.device)

    try:
        if config.save_checkpoints and is_main_process():
            Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        model, optimiser, scheduler = create_model_and_optimizer(config, device)

        start_step = 0
        dataset_state_dict = None

        if args.resume:
            if args.resume == "latest":
                checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
                if checkpoint_path is None:
                    if is_main_process():
                        print("No checkpoint found, starting from scratch")
                else:
                    if is_main_process():
                        print(f"Resuming from latest checkpoint: {checkpoint_path}")
            else:
                checkpoint_path = args.resume

            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                start_step = load_checkpoint(
                    checkpoint_path, model, optimiser, scheduler, device
                )
                dataset_state_dict = checkpoint.get("dataset_state_dict", None)
            elif args.resume != "latest":
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if is_main_process() and config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config),
                resume="allow" if args.resume else None,
            )

        dataloader, eval_dataloader = create_dataloaders(
            config, epoch=0, dataset_state_dict=dataset_state_dict
        )

        if is_main_process() and config.use_wandb:
            watch_model = model.module if isinstance(model, DDP) else model
            wandb.watch(
                watch_model,
                log=config.wandb_watch_log,
                log_freq=config.wandb_watch_log_freq,
            )

        inception_model = create_inception_model(device)

        start_time = time.time()
        print(f"{ start_time= }")
        training_loop(
            model,
            optimiser,
            scheduler,
            dataloader,
            eval_dataloader,
            device,
            inception_model,
            config,
            start_step,
        )
        end_time = time.time()
        print(f"{end_time=}")

        if is_main_process():
            wandb.finish()
    finally:
        if config.distributed:
            cleanup_distributed()
    return end_time - start_time


def main():
    args = parse_args()
    config = ConfigType(args.config).to_config()
    print("Config:")
    pprint(asdict(config))

    if config.distributed:
        world_size = config.world_size
        mp.spawn(
            train_worker,
            args=(world_size, config, args),
            nprocs=world_size,
            join=True,
        )
    else:
        train_worker(0, 1, config, args)


if __name__ == "__main__":
    main()
