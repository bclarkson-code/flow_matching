import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from flow_matching.checkpoint import find_latest_checkpoint, load_model_from_checkpoint
from flow_matching.config import Config, register_configs
from flow_matching.eval_datasets import load_datasets
from flow_matching.metrics import compute_fid


register_configs()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> dict[str, float]:
    """Main evaluation function using Hydra for configuration management.

    Args:
        cfg: Hydra DictConfig containing all configuration parameters

    Usage:
        # Use default config with latest checkpoint
        python evaluate.py

        # Use a specific experiment config
        python evaluate.py +experiment=debug
        python evaluate.py +experiment=full_scale

        # Override specific parameters
        python evaluate.py evaluation.batch_size=128
        python evaluate.py evaluation.images_per_class=100

        # Specify a checkpoint path
        python evaluate.py checkpoint_path=checkpoints/step_10000.pt

        # Combine experiment with overrides
        python evaluate.py +experiment=debug evaluation.batch_size=64
    """
    config: Config = OmegaConf.to_object(cfg)  # type: ignore
    device = cfg.get("device", "cuda")

    checkpoint_path = find_latest_checkpoint(config.checkpoint.checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found in {config.checkpoint.checkpoint_dir}"
        )

    model = load_model_from_checkpoint(checkpoint_path, device=device, config=config)

    scores = {}
    for name, dataset in load_datasets(config).items():
        print(f"Evaluating against: {name}")
        score = compute_fid(model, config, dataset, torch.device("cuda:0"))
        print(f"\n{name}: FID Score: {score:.2f}")
        scores[name] = score
    return scores


if __name__ == "__main__":
    scores = main()
    print(scores)
