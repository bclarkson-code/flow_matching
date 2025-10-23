# Flow Matching for Text-to-Image Generation

A modern, efficient implementation of **Flow Matching** for text-to-image generation. Unlike traditional diffusion models that rely on iterative denoising, flow matching learns continuous flows from noise to data, resulting in faster sampling and more stable training.

This project trains a transformer-based architecture to generate 64x64 images from text descriptions using a 2M image-caption dataset.

## Why Flow Matching?

Flow matching offers several advantages over traditional diffusion models:

- **Simpler objective**: Directly predicts velocity fields instead of noise
- **Faster sampling**: Continuous-time formulation allows for flexible step counts
- **Stable training**: Straightforward MSE loss without complex noise schedules
- **Deterministic inference**: Follows smooth ODE trajectories from noise to image

## Key Features

- **Transformer-based architecture** with cross-attention between image and text tokens
- **Pre-trained embeddings**: Frozen T5-small for text encoding and Stability AI's VAE for image latents
- **Distributed training**: Multi-GPU support via PyTorch DDP (optimized for 2x RTX 3090s)
- **Comprehensive evaluation**: FID scoring using Inception-v3 features
- **Flexible configuration system**: Pre-configured setups for debugging, hyperparameter tuning, and full-scale training
- **Interactive demo**: Gradio web interface with progressive generation visualization

## Architecture

```
Text Input → T5-small Encoder (frozen) → Text Embeddings (128 tokens)
                                              ↓
Random Noise → VAE Encoder (frozen) → Latent Space (8x8x4)
                                              ↓
                                     Patch Embedder (64 tokens)
                                              ↓
                           ┌──────────────────────────────┐
                           │  12 Transformer Blocks        │
                           │  - Cross-attention with text   │
                           │  - AdaLN time conditioning     │
                           │  - Separate image/text MLPs    │
                           └──────────────────────────────┘
                                              ↓
                                    Velocity Prediction
                                              ↓
                            VAE Decoder (frozen) → 64x64 RGB Image
```

**Model specs:**
- Embedding dimension: 768
- Attention heads: 12
- Transformer blocks: 12
- Parameters: ~120M (trainable) + frozen T5 and VAE

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers diffusers datasets accelerate wandb gradio
uv pip install scipy pillow tqdm
```

Or install from a requirements file:

```bash
uv pip install -r wandb/run-*/files/requirements.txt
```

## Quick Start

### 1. Download the dataset

```bash
python load_dataset.py
```

This downloads and preprocesses the [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M) dataset (2 million image-caption pairs resized to 64x64).

### 2. Train the model

**For your first run, use the StabilityConfig** (recommended starting point):

```bash
python train.py --config stability
```

This runs a medium-scale training session with:
- Effective batch size: 576 (96 per GPU × 3 gradient accumulation × 2 GPUs)
- 30,000 steps (~8.5 epochs)
- Evaluation every 1,000 steps
- Checkpoint saving enabled

**Other configurations:**

```bash
# Quick debugging (small dataset subset)
python train.py --config debug

# Check if model can overfit small dataset
python train.py --config overfit

# Hyperparameter tuning (5K steps)
python train.py --config hparam_tuning

# Full-scale training (300K steps)
python train.py --config full_scale
```

**Resume from checkpoint:**

```bash
# Resume from latest checkpoint
python train.py --config stability --resume latest

# Resume from specific checkpoint
python train.py --resume checkpoints/checkpoint_step_10000.pt
```

### 3. Generate images

Launch the interactive Gradio demo:

```bash
python demo.py
```

For public sharing:

```bash
python demo.py --share
```

The demo provides:
- Real-time progressive generation visualization
- Adjustable update intervals
- Example prompts to get you started

## Configuration System

The project includes several pre-configured training setups:

| Config | Purpose | Steps | Batch Size | Use Case |
|--------|---------|-------|------------|----------|
| `stability` | **Recommended first run** | 30K | 576 | Validate stable training over medium duration |
| `overfit` | Sanity check | 5K | 288 | Verify model can memorize small dataset |
| `hparam_tuning` | Quick experiments | 5K | 288 | Dial in learning rates and other hyperparams |
| `debug` | Development | Unlimited | 288 | One-off debugging and testing |
| `full_scale` | Production training | 300K | 576 | Train best possible model |

All configurations support:
- Gradient accumulation for effective large batch sizes
- Automatic checkpoint management (keep recent + periodic checkpoints)
- W&B logging with comprehensive metrics
- Distributed training across multiple GPUs

## Training Details

### Flow Matching Objective

The model learns to predict velocity fields that transform noise into images:

1. Sample a random time `t ∈ [0, 1]`
2. Create noisy latent: `z_t = noise + t * (latent - noise)`
3. Predict velocity: `v_pred = model(z_t, text, t)`
4. Target velocity: `v_target = latent - noise`
5. Minimize MSE: `loss = ||v_pred - v_target||²`

### Inference

Generation follows the learned flow using 50 steps (configurable):

```python
z = random_noise
for t in [0, 1/50, 2/50, ..., 1]:
    v = model(z, text, t)
    z = z + v * (1/50)
image = vae_decode(z)
```

### Learning Rate Schedule

- **Warmup**: Linear increase from 10% of max LR over first 5% of steps
- **Decay**: Cosine annealing to 10% of max LR
- **Max LR**: 5e-4 (tuned via `hparam_tuning` config)

## Results

Training with `StabilityConfig` produces models capable of generating coherent 64x64 images from text prompts after ~30K steps. The model learns to:

- Understand text descriptions and generate corresponding visual content
- Follow color, object, and scene composition instructions
- Generate diverse outputs from similar prompts

**Example prompts:**
- "a photo of a cat"
- "a beautiful sunset over the ocean"
- "a red car on a city street"
- "a painting of a mountain landscape"

*Note: Results depend on training duration and dataset coverage. The 2M dataset provides broad coverage of common objects and scenes.*

## Benchmarking

Test your hardware performance:

```bash
python bench.py
```

This will:
- Find the optimal batch size for your GPU
- Benchmark training speed (images/second)
- Report memory usage

## Project Structure

```
flow_matching/
├── config.py           # Configuration classes and presets
├── model.py            # DiffusionTransformer architecture
├── train.py            # Training loop with distributed support
├── demo.py             # Gradio web interface
├── load_dataset.py     # Dataset download and preprocessing
├── bench.py            # Performance benchmarking
├── debug.py            # Quick dataset inspection
├── checkpoints/        # Saved model checkpoints
└── data/               # Downloaded dataset cache
```

## Technical Deep Dive

### Why These Design Choices?

**Frozen embeddings**: Training text encoders and VAEs from scratch requires enormous compute. By freezing pre-trained T5 and Stability VAE models, we focus compute on learning the image generation transformer.

**Velocity prediction**: Flow matching's velocity formulation is mathematically simpler than diffusion's noise prediction, leading to more stable gradients and easier hyperparameter tuning.

**Transformer architecture**: Cross-attention naturally conditions image generation on text features. AdaLN (Adaptive Layer Normalization) efficiently incorporates time information throughout the network.

**64x64 resolution**: Operating in VAE latent space (8x8x4) is memory efficient. The 64x64 pixel output is a good balance between quality and training speed on consumer GPUs.


## Monitoring and Logging

The project integrates with [Weights & Biases](https://wandb.ai) for experiment tracking:

**Logged metrics:**
- Training/evaluation loss
- FID (Frechet Inception Distance) score
- Learning rate and gradient norms
- Generated images at each evaluation
- Latent and velocity statistics

To disable W&B:

```python
# In config.py
use_wandb: bool = False
```

## Contributing

This is a research/educational project demonstrating modern generative modeling techniques. Potential improvements:

- **Higher resolution**: Train on 256x256 or 512x512 images
- **Better architecture**: Experiment with DiT (Diffusion Transformer) or U-ViT designs
- **Classifier-free guidance**: Add unconditional training for guidance during inference
- **Better sampling**: Implement adaptive step size or higher-order ODE solvers
- **Dataset**: Curate or filter the training data for better quality

## Acknowledgments

- **Flow Matching**: [Flow Matching for Generative Modeling (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747)
- **Architecture inspiration**: [Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748)
- **Pre-trained models**: [T5](https://huggingface.co/t5-small) and [Stability AI VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- **Dataset**: [jackyhate/text-to-image-2M](https://huggingface.co/datasets/jackyhate/text-to-image-2M)

## License

This project is for educational and research purposes. Please check the licenses of the pre-trained models and dataset before commercial use.

---

**Questions or issues?** Open an issue on GitHub or reach out!

Built with PyTorch, Transformers, and modern flow matching techniques.
