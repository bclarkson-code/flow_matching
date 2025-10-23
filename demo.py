import torch
import gradio as gr
import os
from PIL import Image
import torchvision.transforms as transforms
from train import find_latest_checkpoint
from model import DiffusionTransformer
from config import Config


def load_model(checkpoint_path: str, device: torch.device):
    """Load the model from a checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint.get("config", Config())
    model = DiffusionTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


@torch.no_grad()
def generate_images_progressive(
    model: DiffusionTransformer,
    text: list[str],
    config: Config,
    device: torch.device,
    update_interval: int = 5,
):
    """Generate images from text prompts, yielding intermediate results."""
    batch_size = len(text)

    # Create random noise as starting point
    generated_latents = torch.randn(
        batch_size, config.model.latent_channels, 8, 8, device=device
    )

    # Flow matching inference loop
    for t_idx in range(config.num_inference_steps):
        t = torch.ones(batch_size, device=device) * (t_idx / config.num_inference_steps)
        pred_v = model(image_latents=generated_latents, text=text, time=t)
        dt = 1.0 / config.num_inference_steps
        generated_latents = generated_latents + pred_v * dt

        # Yield intermediate results at specified intervals
        if (
            t_idx + 1
        ) % update_interval == 0 or t_idx == config.num_inference_steps - 1:
            # Decode current latents to images
            intermediate_images = model.image_embedder.from_latent(generated_latents)
            intermediate_images = torch.clamp(intermediate_images, 0, 1)

            # Convert to PIL Image
            image = intermediate_images[0].cpu()
            image = transforms.ToPILImage()(image)

            yield image


def create_demo(checkpoint_dir: str = "checkpoints", device: str = "cuda"):
    """Create and launch the Gradio demo interface."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Find and load the latest checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading model from {checkpoint_path}")
    model, config = load_model(checkpoint_path, device)
    print(
        f"Model loaded successfully! Using {config.num_inference_steps} inference steps"
    )

    def generate(prompt: str, update_interval: int = 5):
        """Generate an image from a text prompt, yielding intermediate results."""
        if not prompt.strip():
            return None

        # Generate image with progressive updates
        for image in generate_images_progressive(
            model, [prompt], config, device, update_interval=update_interval
        ):
            yield image

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Markdown("# Text-to-Image Generation")
        gr.Markdown(
            f"Generate images from text prompts using a flow matching diffusion model.\n\n"
            f"**Using checkpoint:** `{os.path.basename(checkpoint_path)}`\n\n"
            f"**Inference steps:** {config.num_inference_steps}"
        )

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter a description of the image you want to generate...",
                    lines=3,
                )
                update_interval = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Update Interval",
                    info="How often to update the image (every N steps)",
                )
                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column():
                output_image = gr.Image(label="Generated Image", type="pil")

        gr.Examples(
            examples=[
                ["a photo of a cat"],
                ["a beautiful sunset over the ocean"],
                ["a red car on a city street"],
                ["a painting of a mountain landscape"],
            ],
            inputs=prompt_input,
        )

        generate_btn.click(
            fn=generate,
            inputs=[prompt_input, update_interval],
            outputs=output_image,
        )

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Gradio demo for text-to-image generation"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public shareable link"
    )
    args = parser.parse_args()

    demo = create_demo(checkpoint_dir=args.checkpoint_dir, device=args.device)
    demo.launch(share=args.share, server_name="0.0.0.0")
