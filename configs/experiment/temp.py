from dataclasses import dataclass, replace

from flow_matching.config import Config


@dataclass
class NewConfig(Config):
    training.learning_rate: float = 1e3

def make_debug_config():
    config = Config()
    config.training.learning_rate = 1e-3
    return config





# --config.seed=0 config.
