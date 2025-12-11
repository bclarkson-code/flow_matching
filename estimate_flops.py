import logging

import hydra
import torch
from fvcore.nn import FlopCountAnalysis
from omegaconf import DictConfig, OmegaConf

from flow_matching.config import Config, register_configs
from flow_matching.dataset import create_eval_dataloader
from flow_matching.model import create_model_and_optimizer


logging.getLogger("fvcore.nn.jit_analysis").setLevel(logging.ERROR)

register_configs()


def estimate_forward_pass_flops(config: Config) -> int:
    config.distributed.distributed = False
    config.model.compile = False

    device = torch.device(config.device)
    model, _, _ = create_model_and_optimizer(config, device)
    eval_dataset = create_eval_dataloader(config, rank=0, world_size=1)

    batch = next(iter(eval_dataset))
    latents, text_embedding, attention_mask = (
        batch["latents"].squeeze(),
        batch["text_embeds"].squeeze(),
        batch["attention_mask"].squeeze(),
    )

    latents = latents.to(device).squeeze(1)
    text_embedding = text_embedding.to(device)
    attention_mask = attention_mask.to(device)

    noise = torch.randn_like(latents).to(device)
    time = torch.rand(len(batch["latents"])).to(device)
    time = time.view(-1, 1, 1, 1)

    noisy_latents = (1 - time) * noise + (time * latents)
    time = time.squeeze()
    inputs = {
        "image_latents": noisy_latents,
        "time": time,
        "text": None,
        "text_embedding": text_embedding,
        "text_mask": attention_mask,
    }

    return FlopCountAnalysis(model, tuple(inputs.values())).total()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> int:
    config: Config = OmegaConf.to_object(cfg)  # type: ignore

    single_pass_flops = estimate_forward_pass_flops(config)

    # backward ~= 2 * forward
    forwards_per_backward = 2

    # might do multiple forwards per 1 back
    single_train_step_flops = single_pass_flops * (
        config.training.gradient_accumulation_steps + forwards_per_backward
    )

    # multiply by number of gpus
    flops_per_turn = single_train_step_flops * config.distributed.world_size

    # multiply by number of steps
    total_flops = flops_per_turn * config.training.num_steps

    print(f"Flops: {total_flops}")
    print(f"TFlops: {total_flops * 1e-12: .2f}")

    return total_flops


if __name__ == "__main__":
    main()
