2025-12-03
 - Ran benchmark with 2 gpus:
    uv run torchrun --nproc-per-node=2 bench.py
    ...
    Rank 0] 2025-12-03 12:34:16,874 - INFO - Final Eval Loss: 1.0347
    Speedrun took: 305.892 seconds
    Speedrun took: 306.089 seconds

 - Ran same benchmark with 1 gpu (ID 0) (but same number of images)
    uv run torchrun --nproc-per-node=1 bench.py
    ...
    Rank 0] 2025-12-03 12:41:19,893 - INFO - Final Eval Loss: 0.9892 
    Speedrun took: 371.434 seconds
 - Ran same on other gpu (ID 1)
    CUDA_VISIBLE_DEVICES=1 uv run torchrun --nproc-per-node=1 bench.py
    ...
    Rank 0] 2025-12-03 12:51:30,884 - INFO - Final Eval Loss: 1.0388 
    Speedrun took: 411.784 seconds

ID 1 is quite a bit slower than ID 0, I think this is due to the very poor
ventilation.

The goal for today is to set up a scaling ladder script. The idea is that 
I can measure the effect of an intervention at different scales. 

I added a ladder rung for the current full scale model and have started 
training:
$ uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_large

2025-12-04
the trainig run failed due to a memory leak. I found that my machine would
lock up when I use up all the ram so I've started runnign debugging like 
this: 
   `systemd-run --user --scope -p MemoryMax=48G uv run torchrun --nproc_per_node=2 train.py +experiment=scaling_ladder_extra_small`
