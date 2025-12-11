import hydra
from omegaconf import DictConfig, OmegaConf

from flow_matching.checkpoint import find_latest_checkpoint, load_checkpoint
from flow_matching.config import Config, register_configs
from train import run_final_evals


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
    config.distributed.distributed = False
    assert checkpoint_path is not None
    _, model, *_ = load_checkpoint(
        checkpoint_path, config, device, include_embedder=True
    )

    return run_final_evals(model, config, device)


if __name__ == "__main__":
    scores = main()
    print(scores)
