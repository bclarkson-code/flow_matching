uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_extra_small &&
uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_small &&
uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_medium &&
uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_large
