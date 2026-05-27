import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch

from audio_flow.encoders.text.t5 import T5
from audio_flow.encoders.text.char import CharEncoder
from audio_flow.adapters.convnext import ConvNeXt

from audio_flow.utils import mean_pool, check_masks_type


def masked_mean(x: Tensor, mask: Tensor, keepdims: bool = False) -> Tensor:
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
    out = (x * mask[:, :, None]).sum(dim=1) / denom
    if keepdims:
        out = out[:, None, :]
    return out


class TTSAdapter(nn.Module): 
    def __init__(self, dim: int, text_dim: int | None = None, **kwargs):
        super().__init__()
        text_dim = text_dim or dim

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # Character encoder
        self.char_encoder = CharEncoder()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, text_dim)
        self.char_conv = ConvNeXt(text_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_fc = nn.Linear(text_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)
        
        # Prompt
        prompt, prompt_mask = self.char_encoder(data["prompt"])  # (b, l_text, d), (b, l_text)
        prompt = self.char_embedder(prompt)  # (b, l_text, d)
        prompt = self.char_conv(prompt, prompt_mask)
        prompt = self.text_norm(prompt)
        prompt = prompt.masked_fill(~prompt_mask[:, :, None], 0.0)
        prompt = self.text_fc(prompt)
        prompt = prompt.masked_fill(~prompt_mask[:, :, None], 0.0)
        text_cond = masked_mean(prompt, prompt_mask, keepdims=True)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)
        cross_attn_mask = prompt_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_v)
        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)

        c = task + text_cond  # Added to timestep embedding in DiT before AdaLN.
        seq = prompt  # (b, l, d)

        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 


class ZeroShotTTSAdapter(nn.Module):
    r"""Condition TTS generation on text and reference speech latents.

    This adapter is intended for F5-TTS-style infilling training. The training
    objective provides `cond_latent`, which is the target speech latent with a
    random span zeroed out. The unmasked latent frames act as the reference
    speech prompt.
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        speaker_dim: int | None = None,
        text_dim: int | None = None,
        **kwargs,
    ):
        super().__init__()
        speaker_dim = speaker_dim or 0
        text_dim = text_dim or dim

        # T5 task encoder
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # Character encoder for transcript text
        self.char_encoder = CharEncoder()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, text_dim)
        self.char_conv = ConvNeXt(text_dim)
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_fc = nn.Linear(text_dim, dim)

        self.learned_speaker_dim = speaker_dim
        if speaker_dim:
            self.speaker_embed = nn.Parameter(torch.randn(1, 1, speaker_dim) * 0.02)

    def forward(
        self,
        data: dict,
        drop_audio_cond: bool = False,
        drop_spk_cond: bool | None = None,
        drop_text: bool = False,
    ) -> dict:
        if drop_spk_cond is None:
            drop_spk_cond = drop_audio_cond

        # Task
        task, task_mask = self.t5(data["task"])
        task = self.t5_fc(task)
        task = mean_pool(task, task_mask, keepdims=True)

        target_mask = data["target_mask"]

        # Reference speech prompt
        cond_latent = data["cond_latent"]
        if drop_audio_cond:
            cond_latent = torch.zeros_like(cond_latent)
        input_cond = cond_latent
        spk_embed = None
        if self.learned_speaker_dim:
            spk_embed = F.normalize(self.speaker_embed, dim=-1)
            spk_embed = spk_embed.expand(target_mask.shape[0], target_mask.shape[1], -1)
            if drop_spk_cond:
                spk_embed = torch.zeros_like(spk_embed)
            spk_embed = spk_embed.masked_fill(~target_mask[:, :, None], 0.0)

        # Transcript prompt
        prompt_ids, prompt_mask = self.char_encoder(data["prompt"])
        prompt = self.char_embedder(prompt_ids)
        prompt = self.char_conv(prompt, prompt_mask)
        prompt = self.text_norm(prompt)
        prompt = prompt.masked_fill(~prompt_mask[:, :, None], 0.0)
        prompt = self.text_fc(prompt)
        prompt = prompt.masked_fill(~prompt_mask[:, :, None], 0.0)
        if drop_text:
            prompt = torch.zeros_like(prompt)
            prompt_mask = torch.zeros_like(prompt_mask)
        text_cond = masked_mean(prompt, prompt_mask, keepdims=True)

        seq = prompt
        seq_mask = prompt_mask

        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]
        cross_attn_mask = seq_mask[:, None, None, :] * target_mask[:, None, :, None]
        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)

        B, target_len = target_mask.shape
        text_len = prompt_mask.shape[1]
        device = target_mask.device
        dtype = prompt.dtype
        target_lengths = target_mask.sum(dim=1).clamp(min=1).to(dtype)
        prompt_lengths = prompt_mask.sum(dim=1).clamp(min=1).to(dtype)
        audio_last_pos = (target_lengths - 1).clamp(min=0)
        text_denominator = (prompt_lengths - 1).clamp(min=1)

        audio_pos = torch.arange(target_len, device=device, dtype=dtype)[None, :].expand(B, -1)
        text_unit = audio_last_pos[:, None] / text_denominator[:, None]
        text_pos = torch.arange(text_len, device=device, dtype=dtype)[None, :] * text_unit
        text_pos = text_pos.masked_fill(~prompt_mask, 0.0)
        cross_k_pos = text_pos

        controls = {
            "c": task + text_cond,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask,
            "cross_q_pos": audio_pos,
            "cross_k_pos": cross_k_pos,
            "input_cond": input_cond,
        }
        if spk_embed is not None:
            controls["spk_embed"] = spk_embed
        return controls


class ZeroShotTTSSpeakerAdapter(ZeroShotTTSAdapter):
    r"""Zero-shot TTS adapter with utterance-level speaker embedding concat.

    The speaker embedding is repeated to the audio latent length and passed as
    `controls["input_speaker"]`, so the DiT input projection can consume
    `[x_t, input_cond, input_speaker]` frame by frame.
    """

    def __init__(
        self,
        in_dim: int,
        dim: int,
        text_dim: int | None = None,
        speaker_dim: int = 256,
        speaker_embedding_key: str = "speaker_embedding",
        **kwargs,
    ):
        super().__init__(in_dim=in_dim, dim=dim, speaker_dim=None, text_dim=text_dim, **kwargs)
        self.speaker_dim = speaker_dim
        self.speaker_embedding_key = speaker_embedding_key

    def forward(
        self,
        data: dict,
        drop_audio_cond: bool = False,
        drop_spk_cond: bool | None = None,
        drop_text: bool = False,
    ) -> dict:
        if drop_spk_cond is None:
            drop_spk_cond = drop_audio_cond
        controls = super().forward(
            data,
            drop_audio_cond=drop_audio_cond,
            drop_spk_cond=drop_spk_cond,
            drop_text=drop_text,
        )

        if self.speaker_embedding_key not in data:
            raise KeyError(
                f"`{self.speaker_embedding_key}` is required by ZeroShotTTSSpeakerAdapter. "
                "Set dataset.speaker_embedding_root in the config or pass the tensor during sampling."
            )

        speaker_embedding = data[self.speaker_embedding_key].to(
            device=data["target_mask"].device,
            dtype=controls["input_cond"].dtype,
        )
        if speaker_embedding.dim() != 2:
            raise ValueError(
                f"`{self.speaker_embedding_key}` must have shape (B, D), "
                f"got {tuple(speaker_embedding.shape)}."
            )
        if speaker_embedding.shape[-1] != self.speaker_dim:
            raise ValueError(
                f"`{self.speaker_embedding_key}` dim mismatch: expected {self.speaker_dim}, "
                f"got {speaker_embedding.shape[-1]}."
            )

        target_mask = data["target_mask"]
        input_speaker = speaker_embedding[:, None, :].expand(-1, target_mask.shape[1], -1)
        input_speaker = input_speaker.masked_fill(~target_mask[:, :, None], 0.0)
        if drop_spk_cond:
            input_speaker = torch.zeros_like(input_speaker)

        controls["input_speaker"] = input_speaker
        return controls
