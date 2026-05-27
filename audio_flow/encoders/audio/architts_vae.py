from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.package import PackageImporter


DEFAULT_REPO_ID = "archimickey/architts-vae12_5hz"
CKPT_FILENAME = "architts_vae12_5hz.pt"


def resolve_architts_vae_ckpt(ckpt_path: str | Path | None = None, repo_id: str = DEFAULT_REPO_ID) -> Path:
    if ckpt_path is not None:
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    repo_root = Path(__file__).resolve().parents[3]
    local_candidates = [
        repo_root / "checkpoints/architts_vae12_5hz" / CKPT_FILENAME,
        repo_root / "checkpoints" / CKPT_FILENAME,
        Path(__file__).resolve().with_name(CKPT_FILENAME),
    ]
    for path in local_candidates:
        if path.exists():
            return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            f"Could not find {CKPT_FILENAME} locally, and huggingface_hub is not installed."
        ) from exc
    return Path(hf_hub_download(repo_id=repo_id, filename=CKPT_FILENAME))


class ArchiTTSVAE(nn.Module):
    r"""AudioFlow wrapper for the packaged ArchiTTS 24 kHz VAE.

    The checkpoint is a torch.package archive that contains the ArchiTTS VAE
    source, resolved config, and weights, so this class does not import the
    external ArchiTTS repository at runtime.
    """

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        vae_name: str = "vae_24khz_f1920c64_1.0",
        repo_id: str = DEFAULT_REPO_ID,
    ):
        super().__init__()
        if vae_name != "vae_24khz_f1920c64_1.0":
            raise ValueError(f"Unsupported packaged ArchiTTS VAE: {vae_name}")

        self.ckpt_path = resolve_architts_vae_ckpt(ckpt_path, repo_id=repo_id)
        importer = PackageImporter(str(self.ckpt_path))
        package = importer.load_pickle("architts_vae", "checkpoint.pkl")
        autoencoder_module = importer.import_module("architts.model.vae.autoencoder")
        pretransform_module = importer.import_module("architts.model.vae.pretransform")

        self.metadata = package["metadata"]
        self.model = pretransform_module.AutoencoderPretransform(
            autoencoder_module.create_autoencoder_from_config(package["config"]),
            scale=float(self.metadata["scale"]),
        )
        self.model.load_state_dict(package["state_dict"])
        self.model.requires_grad_(False).eval()

        self.dim = int(self.metadata["encoded_channels"])
        self.sr = int(self.metadata["sample_rate"])
        self.sample_rate = self.sr
        self.fps = float(self.metadata["fps"])
        self.downsampling_ratio = int(self.metadata["downsampling_ratio"])

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

    @torch.inference_mode()
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

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError(f"Expected latent shape (B, T, D), got {tuple(latent.shape)}")
        latent = latent.transpose(1, 2).contiguous()
        return self.model.decode(latent)
