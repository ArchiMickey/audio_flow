from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.models.attention import Block
from audio_flow.models.rope import RoPE


class F5SinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor, scale: float = 1000.0) -> Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class F5TimestepEmbedding(nn.Module):
    r"""F5-TTS timestep embedding: sinusoidal time features followed by an MLP."""

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = F5SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: Tensor) -> Tensor:
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        return self.time_mlp(time_hidden)


class FinalAdaLayerNorm(nn.Module):
    r"""Final AdaLN used before the output projection, following DiT practice."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        if c.dim() == 3:
            c = c.squeeze(1)
        shift, scale = self.linear(c).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class F5StyleCrossAttnTransformer(nn.Module):
    r"""F5-TTS-style DiT backbone with AudioFlow cross-attention text conditioning.

    This keeps the existing zero-shot TTS adapter contract for text/reference
    cross-attention and fractional RoPE positions. The audio stream consumes
    noised audio `x` and masked conditioning audio `controls["input_cond"]`,
    then uses final AdaLN and zero-initialized output projection.
    """

    def __init__(
        self,
        in_dim: int = 16,
        dim: int = 384,
        mlp_ratio: float = 4.0,
        num_layers: int = 12,
        num_heads: int = 12,
        speaker_dim: int = 0,
        rope_len: int = 8192,
        zero_init_output: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.speaker_dim = speaker_dim
        self.fc_in = nn.Linear(in_dim * 2 + speaker_dim, dim)

        self.t_embedder = F5TimestepEmbedding(dim=dim, freq_embed_dim=256)
        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

        self.norm_out = FinalAdaLayerNorm(dim)
        self.fc_out = nn.Linear(dim, in_dim)

        if zero_init_output:
            nn.init.constant_(self.norm_out.linear.weight, 0.0)
            nn.init.constant_(self.norm_out.linear.bias, 0.0)
            nn.init.constant_(self.fc_out.weight, 0.0)
            nn.init.constant_(self.fc_out.bias, 0.0)

    def forward(
        self,
        t: Tensor,
        x: Tensor,
        controls: dict,
        **kwargs,
    ) -> Tensor:
        c = controls["c"]
        seq = controls["seq"]
        self_attn_mask = controls["self_attn_mask"]
        cross_attn_mask = controls["cross_attn_mask"]
        cross_q_pos = controls.get("cross_q_pos")
        cross_k_pos = controls.get("cross_k_pos")

        input_cond = controls.get("input_cond")
        if input_cond is None:
            input_cond = torch.zeros_like(x)
        else:
            input_cond = input_cond.to(device=x.device, dtype=x.dtype)
            if input_cond.shape[1] < x.shape[1]:
                input_cond = nn.functional.pad(input_cond, (0, 0, 0, x.shape[1] - input_cond.shape[1]))
            input_cond = input_cond[:, : x.shape[1], :]

        input_speaker = controls.get("input_speaker")
        if input_speaker is None:
            input_speaker = controls.get("spk_embed")
        if self.speaker_dim:
            if input_speaker is None:
                raise KeyError(
                    "F5StyleCrossAttnTransformer was configured with speaker_dim but controls lack "
                    "input_speaker/spk_embed."
                )
            input_speaker = input_speaker.to(device=x.device, dtype=x.dtype)
            if input_speaker.shape[-1] != self.speaker_dim:
                raise ValueError(f"input_speaker dim mismatch: expected {self.speaker_dim}, got {input_speaker.shape[-1]}.")
            if input_speaker.shape[1] < x.shape[1]:
                input_speaker = nn.functional.pad(input_speaker, (0, 0, 0, x.shape[1] - input_speaker.shape[1]))
            input_speaker = input_speaker[:, : x.shape[1], :]

        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        c = c + self.t_embedder(t)[:, None, :]

        target_mask = self_attn_mask[:, 0, 0, :].bool()
        inputs = [x, input_cond]
        if self.speaker_dim:
            inputs.append(input_speaker)
        x = self.fc_in(torch.cat(inputs, dim=-1))

        for block in self.blocks:
            x = block(x, c, seq, self.rope, self_attn_mask, cross_attn_mask, cross_q_pos, cross_k_pos)

        x = self.norm_out(x, c)
        x = self.fc_out(x)
        x = torch.where(target_mask[:, :, None], x, torch.zeros_like(x))
        return x
