from __future__ import annotations

import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml


def _resolve_config(value, root):
    if isinstance(value, dict):
        return {key: _resolve_config(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_config(item, root) for item in value]
    if not isinstance(value, str):
        return value

    def get_path(path: str):
        cur = root
        for part in path.split("."):
            cur = cur[part]
        return cur

    full_match = re.fullmatch(r"\$\{([^}]+)\}", value)
    if full_match:
        return get_path(full_match.group(1))
    return re.sub(r"\$\{([^}]+)\}", lambda match: str(get_path(match.group(1))), value)


class ArchiTTSVAE(nn.Module):
    r"""Adapter that exposes the ArchiTTS VAE through AudioFlow's VAE API."""

    def __init__(
        self,
        architts_root: str | Path = "/home/archimickey/Projects/ArchiTTS",
        vae_name: str = "vae_24khz_f1920c64_1.0",
    ):
        super().__init__()
        architts_root = Path(architts_root)
        sys.path.insert(0, str(architts_root.resolve()))

        from architts.model.vae.autoencoder import create_autoencoder_from_config
        from architts.model.vae.pretransform import AutoencoderPretransform

        vae_dir = architts_root / "pretrained_models" / vae_name
        config_path = vae_dir / "config.yaml"
        ckpt_path = vae_dir / "model.ckpt"

        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        config = _resolve_config(raw_config, raw_config)

        scale = float(vae_name.split("_")[-1])
        self.model = AutoencoderPretransform(create_autoencoder_from_config(config), scale=scale)
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.requires_grad_(False).eval()

        self.dim = int(self.model.encoded_channels)
        self.sr = int(self.model.sample_rate)
        self.fps = float(self.model.sample_rate) / float(self.model.downsampling_ratio)
        self.downsampling_ratio = int(self.model.downsampling_ratio)

    def _encode_posterior_mean(self, audio: torch.Tensor) -> torch.Tensor:
        autoencoder = self.model.model
        if self.model.model_half:
            audio = audio.half()
            autoencoder.to(torch.float16)

        latent = autoencoder.encoder(audio) if autoencoder.encoder is not None else audio
        bottleneck = autoencoder.bottleneck
        if bottleneck is not None:
            bottleneck_name = type(bottleneck).__name__
            if bottleneck_name == "VAEBottleneck2":
                latent = latent[:, :-1, :]
            elif bottleneck_name == "VAEBottleneck":
                latent = latent.chunk(2, dim=1)[0]
            else:
                latent = bottleneck.encode(latent)

        if self.model.model_half:
            latent = latent.float()
        return latent / self.model.scale

    def encode(self, audio: torch.Tensor, sample_posterior: bool = False) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"Expected audio shape (B, C, T), got {tuple(audio.shape)}")
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)
        remainder = audio.shape[-1] % self.downsampling_ratio
        if remainder:
            audio = F.pad(audio, (0, self.downsampling_ratio - remainder))
        if sample_posterior:
            latent = self.model.encode(audio)
        else:
            latent = self._encode_posterior_mean(audio)
        return latent.transpose(1, 2).contiguous()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent shape (B, T, D), got {tuple(latent.shape)}")
        latent = latent.transpose(1, 2).contiguous()
        return self.model.decode(latent)
