import torch
import math
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from config import Config


class TextEmbedder(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(
            config.model.text_embed_model_string
        )
        self.text_encoder.eval()
        self.tokenizer = T5Tokenizer.from_pretrained(
            config.model.text_embed_model_string, legacy=True
        )
        self.max_length = config.model.text_max_length

    def forward(self, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            tokens = {k: v.to(self.text_encoder.device) for k, v in tokens.items()}
            text_embeddings = self.text_encoder(tokens["input_ids"]).last_hidden_state
            return text_embeddings, tokens["attention_mask"]


class PatchEmbedder(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.proj = torch.nn.Linear(
            config.model.latent_channels, config.model.embedding_dim
        )

        self.pos_embed = torch.nn.Parameter(
            torch.randn(1, config.model.n_image_tokens, config.model.embedding_dim)
            * config.model.patch_embed_init
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x + self.pos_embed

        return x


class ImageEmbedder(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(config.model.image_embed_model_string)
        self.vae.eval()
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        self.patch_embedder = PatchEmbedder(config)
        self.scale_factor = config.model.vae_scale_factor

    def to_latent(self, image: torch.Tensor):
        with torch.no_grad():
            return self.vae.encode(image).latent_dist.mode() * self.scale_factor

    def from_latent(self, latents: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.vae.decode(latents / self.scale_factor).sample

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        embedding = self.patch_embedder(latents)
        return embedding


class TimeEmbedder(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.embedding_dim = config.model.time_embedding_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_dim, self.embedding_dim * 4),
            torch.nn.SiLU(),
            torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim),
        )

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        embedding = math.log(10000) / (half_dim - 1)
        embedding = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -embedding
        )
        embedding = timesteps[:, None] * embedding[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        return self.mlp(embedding)


class AdaLNModulation(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.linear = torch.nn.Linear(
            config.model.time_embedding_dim, config.model.embedding_dim * 2
        )

    def forward(self, x: torch.Tensor, time_embed: torch.Tensor):
        scale, shift = self.linear(time_embed).chunk(2, dim=-1)
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DiffusionTransformerBlock(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_image_tokens = config.model.n_image_tokens
        self.n_text_tokens = config.model.n_text_tokens
        self.embedding_dim = config.model.embedding_dim
        self.time_embed_dim = config.model.time_embedding_dim

        self.attn = torch.nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config.model.n_heads,
            batch_first=True,
        )

        self.mlp_image = torch.nn.Sequential(
            torch.nn.Linear(
                self.embedding_dim, self.embedding_dim * config.model.expansion_factor
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                self.embedding_dim * config.model.expansion_factor, self.embedding_dim
            ),
        )
        self.mlp_text = torch.nn.Sequential(
            torch.nn.Linear(
                self.embedding_dim, self.embedding_dim * config.model.expansion_factor
            ),
            torch.nn.GELU(),
            torch.nn.Linear(
                self.embedding_dim * config.model.expansion_factor, self.embedding_dim
            ),
        )

        self.adaLN_modulation = AdaLNModulation(config)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        time_embed: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = torch.cat([image_tokens, text_tokens], dim=1)
        embedding = self.adaLN_modulation(embedding, time_embed)

        attention, _ = self.attn(
            embedding,
            embedding,
            embedding,
            key_padding_mask=(~attention_mask.bool()),
        )

        embedding = embedding + attention

        embedding = self.adaLN_modulation(embedding, time_embed)

        image_embedding = embedding[:, : self.n_image_tokens, :]
        text_embedding = embedding[:, self.n_image_tokens :, :]

        image_embedding = image_embedding + self.mlp_image(image_embedding)
        text_embedding = text_embedding + self.mlp_text(text_embedding)

        return image_embedding, text_embedding


class DiffusionTransformer(torch.nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = config.model.embedding_dim

        self.image_embedder = ImageEmbedder(config)
        for parameter in self.image_embedder.vae.parameters():
            parameter.requires_grad = False

        self.text_embedder = TextEmbedder(config)
        for parameter in self.text_embedder.parameters():
            parameter.requires_grad = False

        self.time_embedder = TimeEmbedder(config)
        self.blocks = torch.nn.ModuleList(
            [DiffusionTransformerBlock(config) for _ in range(config.model.n_blocks)]
        )
        self.final_layer = torch.nn.Linear(
            self.embedding_dim, config.model.latent_channels
        )
        torch.nn.init.xavier_uniform_(
            self.final_layer.weight, gain=config.model.final_layer_init_gain
        )
        torch.nn.init.zeros_(self.final_layer.bias)

    def forward(self, image_latents: torch.Tensor, text: list[str], time: torch.Tensor):
        image_embedding = self.image_embedder(image_latents)
        text_embedding, text_mask = self.text_embedder(text)
        time_embedding = self.time_embedder(time)

        batch_size = image_embedding.shape[0]
        image_mask = torch.ones(
            batch_size,
            image_embedding.shape[1],
            device=image_embedding.device,
            dtype=torch.bool,
        )
        mask = torch.cat([image_mask, text_mask.bool()], dim=1)

        for block in self.blocks:
            image_embedding, text_embedding = block(
                image_embedding, text_embedding, time_embedding, mask
            )

        batch = image_embedding.shape[0]

        image_embedding = image_embedding.transpose(1, 2).reshape(
            batch, self.embedding_dim, 8, 8
        )
        velocity = self.final_layer(image_embedding.permute(0, 2, 3, 1))
        velocity = velocity.permute(0, 3, 1, 2)

        return velocity
