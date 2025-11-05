import logging
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import torch
import webdataset as wds
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import wandb
from flow_matching.checkpoint import (
    cleanup_old_checkpoints,
    resume_from_checkpoint,
    save_checkpoint,
)
from flow_matching.config import Config, is_committed, register_configs
from flow_matching.dataset import create_eval_dataset, create_train_dataset
from flow_matching.distributed import (
    cleanup_distributed,
    is_main_process,
    setup_distributed,
)
from flow_matching.model import DiffusionTransformer, create_model_and_optimizer


logger = logging.getLogger()


def generate_images(
    model: DiffusionTransformer,
    latents: torch.Tensor,
    config: Config,
    device: torch.device,
    text: list[str] | None = None,
    text_embedding: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        batch_size = latents.shape[0]
        generated_latents = torch.randn_like(latents)

        for t_idx in range(config.logging.num_inference_steps):
            t = torch.ones(batch_size, device=device) * (
                t_idx / config.logging.num_inference_steps
            )
            pred_v = model(
                image_latents=generated_latents,
                text=text,
                text_embedding=text_embedding,
                text_mask=attention_mask,
                time=t,
            )
            dt = 1.0 / config.logging.num_inference_steps
            generated_latents = generated_latents + pred_v * dt

        generated_images = model.image_embedder.from_latent(generated_latents)  # type: ignore
        return torch.clamp(generated_images, 0, 1)


def compute_loss(
    model: DiffusionTransformer | DDP,
    image_latents: torch.Tensor,
    device: torch.device,
    text: list[str] | None = None,
    text_embedding: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    batch_size = image_latents.shape[0]

    with torch.profiler.record_function("preprocess_inputs"):
        noise = torch.randn_like(image_latents).to(device)
        time = torch.rand(batch_size).to(device)
        time = time.view(-1, 1, 1, 1)

        target_velocity = image_latents - noise
        noisy_latents = (1 - time) * noise + (time * image_latents)
        time = time.squeeze()

    with torch.profiler.record_function("forward"):
        predicted_velocity = model(
            image_latents=noisy_latents,
            text=text,
            text_embedding=text_embedding,
            text_mask=attention_mask,
            time=time,
        )
    recovered_latents = (
        noisy_latents + (1 - time.view(-1, 1, 1, 1)) * predicted_velocity
    )

    with torch.profiler.record_function("loss"):
        loss = torch.nn.functional.mse_loss(predicted_velocity, target_velocity)
    return loss, {
        "image_latents": image_latents,
        "noise": noise,
        "time": time,
        "noisy_latents": noisy_latents,
        "predicted_velocity": predicted_velocity,
        "target_velocity": target_velocity,
    }


def evaluate(
    model: DiffusionTransformer | DDP,
    eval_dataset: wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue]
    device: torch.device,
    step: int,
    config: Config,
) -> float:
    if not is_main_process() or config.dataset.eval_samples is None:
        return float("infinity")

    model.eval()
    with torch.no_grad():
        unwrapped_model = model.module if isinstance(model, DDP) else model

        eval_losses = []
        generated_images_list = []
        text_input = []
        n_batches = math.ceil(config.dataset.eval_samples / config.training.batch_size)

        for batch in tqdm(
            eval_dataset, desc="Evaluating", position=1, leave=False, total=n_batches
        ):
            latents, text_embedding, attention_mask = (
                batch["latents"].squeeze(),
                batch["text_embeds"].squeeze(),
                batch["attention_mask"].squeeze(),
            )

            latents = latents.to(device).squeeze(1)
            text_embedding = text_embedding.to(device)
            attention_mask = attention_mask.to(device)

            loss, _ = compute_loss(
                model=unwrapped_model,
                image_latents=latents,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
                device=device,
            )
            eval_losses.append(loss.item())

            generated_images = generate_images(
                model=unwrapped_model,
                latents=latents,
                text_embedding=text_embedding,
                attention_mask=attention_mask,
                config=config,
                device=device,
            )

            generated_images_list.append(generated_images.cpu().numpy())

        avg_eval_loss = float(np.mean(eval_losses))

    if config.logging.use_wandb:
        imgs = np.concatenate(generated_images_list)[
            : config.logging.num_images_to_upload
        ]
        images_for_wandb = [wandb.Image(np.transpose(img, (1, 2, 0))) for img in imgs]
        wandb.log(
            {
                "eval/loss": avg_eval_loss,
                "eval/generated_images": images_for_wandb,
                "eval/text_input": text_input,
            },
            step=step,
        )

    model.train()
    return avg_eval_loss


def log_training_metrics(
    loss: float,
    grad_norm: torch.Tensor,
    optimiser: torch.optim.AdamW,
    step: int,
    config: Config,
    image_latents: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
    time: torch.Tensor | None = None,
    noisy_latents: torch.Tensor | None = None,
    predicted_velocity: torch.Tensor | None = None,
    target_velocity: torch.Tensor | None = None,
) -> None:
    if not is_main_process() or not config.logging.use_wandb:
        return
    metrics = {
        "train/loss": loss,
        "train/grad_norm": grad_norm.item(),
        "train/learning_rate": optimiser.param_groups[0]["lr"],
    }
    if image_latents is not None:
        metrics.update(
            {
                "train/latents_min": image_latents.min().item(),
                "train/latents_mean": image_latents.mean().item(),
                "train/latents_max": image_latents.max().item(),
                "train/latents_std": image_latents.std().item(),
            }
        )
    if noise is not None:
        metrics.update(
            {
                "train/noise_min": noise.min().item(),
                "train/noise_mean": noise.mean().item(),
                "train/noise_max": noise.max().item(),
                "train/noise_std": noise.std().item(),
            }
        )
    if time is not None:
        metrics.update(
            {
                "train/time_min": time.min().item(),
                "train/time_mean": time.mean().item(),
                "train/time_max": time.max().item(),
                "train/time_std": time.std().item(),
            }
        )
    if noisy_latents is not None:
        metrics.update(
            {
                "train/noisy_latents_min": noisy_latents.min().item(),
                "train/noisy_latents_mean": noisy_latents.mean().item(),
                "train/noisy_latents_max": noisy_latents.max().item(),
                "train/noisy_latents_std": noisy_latents.std().item(),
            }
        )
    if predicted_velocity is not None:
        metrics.update(
            {
                "train/predicted_velocity_min": predicted_velocity.min().item(),
                "train/predicted_velocity_mean": predicted_velocity.mean().item(),
                "train/predicted_velocity_max": predicted_velocity.max().item(),
                "train/predicted_velocity_std": predicted_velocity.std().item(),
            }
        )
    if target_velocity is not None:
        metrics.update(
            {
                "train/target_velocity_min": target_velocity.min().item(),
                "train/target_velocity_mean": target_velocity.mean().item(),
                "train/target_velocity_max": target_velocity.max().item(),
                "train/target_velocity_std": target_velocity.std().item(),
            }
        )
    wandb.log(
        metrics,
        step=step,
    )


def train_step(
    model: DiffusionTransformer | DDP,
    dataset: iter,  # type: ignore
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    device: torch.device,
    step: int,
    config: Config,
) -> tuple[DiffusionTransformer | DDP, torch.Tensor]:
    optimiser.zero_grad()

    total_loss = torch.tensor(0.0, device=device)
    with torch.profiler.record_function("load_data"):
        batch = next(dataset)

    for _ in range(config.training.gradient_accumulation_steps):
        with torch.profiler.record_function("preprocess_data"):
            latents, text_embedding, attention_mask = (
                batch["latents"].squeeze(),
                batch["text_embeds"].squeeze(),
                batch["attention_mask"].squeeze(),
            )
            latents = latents.to(device)
            text_embedding = text_embedding.to(device)
            attention_mask = attention_mask.to(device)

        loss, tensors = compute_loss(
            model=model,
            image_latents=latents,
            text_embedding=text_embedding,
            attention_mask=attention_mask,
            device=device,
        )
        with torch.profiler.record_function("load_data"):
            batch = next(dataset)

        loss = loss / config.training.gradient_accumulation_steps

        with torch.profiler.record_function("backward"):
            loss.backward()

        with torch.profiler.record_function("loss"):
            total_loss += loss.detach()

    with torch.profiler.record_function("step"):
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.training.gradient_clip_max_norm
        )

        optimiser.step()
        scheduler.step()

    with torch.profiler.record_function("log"):
        log_training_metrics(
            loss=total_loss.item(),
            grad_norm=grad_norm,
            optimiser=optimiser,
            step=step,
            config=config,
        )

    return model, total_loss


def handle_step_logging_and_checkpointing(
    step: int,
    loss: torch.Tensor,
    model: DiffusionTransformer | DDP,
    eval_dataset: wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue]
    device: torch.device,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    config: Config,
) -> None:
    should_eval = (
        step % config.logging.log_every == 0 and step > 0 and is_main_process()
    )
    if not should_eval:
        return

    logger.info(f"Step {step}, Loss: {loss:.4f}")

    eval_loss = evaluate(
        model=model,
        eval_dataset=eval_dataset,
        device=device,
        step=step,
        config=config,
    )
    logger.info(f"Step {step}, Eval Loss: {eval_loss:.4f}")

    if not (
        config.checkpoint.save_checkpoints
        and step % config.checkpoint.checkpoint_freq == 0
        and step > 0
    ):
        return None

    cleanup_old_checkpoints(config.checkpoint.checkpoint_dir, config)
    save_checkpoint(
        model=model,
        optimiser=optimiser,
        scheduler=scheduler,
        step=step,
        loss=loss.item(),
        config=config,
    )


def training_loop(
    model: DiffusionTransformer | DDP,
    optimiser: torch.optim.AdamW,
    scheduler: torch.optim.lr_scheduler.SequentialLR,
    dataset: wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue]
    eval_dataset: wds.WebDataset,  # pyright: ignore[reportAttributeAccessIssue]
    device: torch.device,
    config: Config,
    start_step: int = 0,
) -> None:
    step = start_step

    pbar_context = tqdm(
        initial=start_step,
        total=config.training.num_steps,
        desc="Training",
        position=0,
        disable=not is_main_process(),
    )
    dataset = iter(dataset)

    with pbar_context as pbar:
        while step < config.training.num_steps:
            model, loss = train_step(
                model,
                dataset,
                optimiser,
                scheduler,
                device,
                step,
                config,
            )

            handle_step_logging_and_checkpointing(
                step=step,
                loss=loss,
                model=model,
                eval_dataset=eval_dataset,
                device=device,
                optimiser=optimiser,
                scheduler=scheduler,
                config=config,
            )

            step += 1
            pbar.update(1)

    eval_loss = evaluate(
        model=model,
        eval_dataset=eval_dataset,
        device=device,
        step=step,
        config=config,
    )
    if is_main_process():
        logger.info(f"Final Eval Loss: {eval_loss:.4f}")


def train_worker(
    rank: int, world_size: int, config: Config, resume_path: str | None = None
) -> float:
    torch.set_float32_matmul_precision("high")
    if config.distributed.distributed:
        setup_distributed(rank, world_size, config)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device(config.device)

    try:
        if config.checkpoint.save_checkpoints and is_main_process():
            Path(config.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        start_step: int = 0

        if resume_path:
            model, optimiser, scheduler, train_dataset, eval_dataset = (
                resume_from_checkpoint(resume_path, config=config, device=device)
            )
        else:
            model, optimiser, scheduler = create_model_and_optimizer(config, device)
            train_dataset = create_train_dataset(config=config)
            eval_dataset = create_eval_dataset(config=config)

        if is_main_process() and config.logging.use_wandb:
            wandb.init(
                project=config.logging.wandb_project,
                config=asdict(config),
                resume="allow" if resume_path else None,
            )

        start_time = time.time()
        training_loop(
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            device=device,
            config=config,
            start_step=start_step,
        )
        end_time = time.time()

        if is_main_process():
            wandb.finish()
    finally:
        if config.distributed.distributed:
            cleanup_distributed()
    return end_time - start_time


register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration management.

    Args:
        cfg: Hydra DictConfig containing all configuration parameters

    Usage:
        # Use default config
        python train.py

        # Use a specific experiment config (defined in ConfigStore)
        python train.py +experiment=debug
        python train.py +experiment=overfit
        python train.py +experiment=stability
        python train.py +experiment=hparam_tuning
        python train.py +experiment=full_scale

        # Override specific parameters
        python train.py training.batch_size=256 training.learning_rate=1e-3

        # Override nested config parameters
        python train.py dataset.num_workers=16 logging.use_wandb=false

        # Resume from checkpoint
        python train.py resume_path=checkpoints/latest.pt

        # Combine experiment with overrides
        python train.py +experiment=debug training.num_steps=1000
    """
    config: Config = OmegaConf.to_object(cfg)  # type: ignore
    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}")

    resume_path = cfg.get("resume_path", None)

    if config.distributed.distributed:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        logger.info(f"Rank {rank}/{world_size}, Local rank: {local_rank}")
        train_worker(local_rank, world_size, config, resume_path)
    else:
        train_worker(0, 1, config, resume_path)


if __name__ == "__main__":
    if not is_committed():
        raise ValueError(
            "Changes must be commited and pushed before experiments can be run"
        )
    main()
