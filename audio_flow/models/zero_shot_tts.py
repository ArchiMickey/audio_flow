from __future__ import annotations

from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher


class ZeroShotTTSFlowMatcher(nn.Module):
    r"""Zero-shot TTS training wrapper using AudioFlow's CFM backend.

    The model trains by masking a random contiguous span of the target speech
    latent. The unmasked target frames become the reference speech condition,
    and the loss is only computed on the masked span.

    Flow matching itself is delegated to `ConditionalFlowMatcher`, the same
    class used by `train.py` for the existing AudioFlow models. This class only
    adds the zero-shot TTS training framework around it: span masking,
    classifier-free condition dropout, and reference-latent conditioning.
    """

    skip_default_validate = True

    def __init__(
        self,
        base: nn.Module,
        adapter: nn.Module,
        sigma: float = 0.0,
        audio_drop_prob: float = 0.3,
        cond_drop_prob: float = 0.2,
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        **kwargs,
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.sigma = sigma
        self.fm = ConditionalFlowMatcher(sigma=sigma)
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        self.frac_lengths_mask = frac_lengths_mask

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def compute_loss(self, data: dict) -> dict[str, Tensor]:
        x1 = data["target_latent"]
        target_mask = data["target_mask"].bool()

        span_mask = self._random_span_mask(target_mask)

        x0 = torch.randn_like(x1)
        time, xt, flow = self.fm.sample_location_and_conditional_flow(x0=x0, x1=x1)

        cond_latent = torch.where(span_mask[:, :, None], torch.zeros_like(x1), x1)
        cond_latent = torch.where(target_mask[:, :, None], cond_latent, torch.zeros_like(cond_latent))
        cond_mask = target_mask & ~span_mask

        drop_audio_cond = False
        drop_spk_cond = False
        if random() < self.cond_drop_prob:
            drop_audio_cond = True
            drop_spk_cond = True
            drop_text = True
        else:
            drop_text = False
            if random() < self.audio_drop_prob:
                if random() < 0.5:
                    drop_audio_cond = True
                    drop_spk_cond = True
                elif random() < 0.5:
                    drop_audio_cond = True
                else:
                    drop_spk_cond = True

        cond_data = dict(data)
        cond_data["cond_latent"] = cond_latent
        cond_data["cond_mask"] = cond_mask

        controls = self.adapter(
            cond_data,
            drop_audio_cond=drop_audio_cond,
            drop_spk_cond=drop_spk_cond,
            drop_text=drop_text,
        )
        pred = self.base(t=time, x=xt, controls=controls)

        loss_per_frame = ((pred - flow) ** 2).mean(dim=-1)
        loss = loss_per_frame[span_mask].mean()

        return {
            "loss": loss,
            "span_mask": span_mask,
            "cond_latent": cond_latent,
            "pred": pred,
        }

    def forward(self, data: dict) -> dict[str, Tensor]:
        return self.compute_loss(data)

    @torch.no_grad()
    def sample(
        self,
        data: dict | None = None,
        cond_latent: Tensor | None = None,
        prompt: list[str] | str | None = None,
        duration: int | Tensor | None = None,
        *,
        lens: Tensor | None = None,
        steps: int = 32,
        cfg_strength: float = 1.0,
        sway_sampling_coef: float | None = None,
        seed: int | None = None,
        max_duration: int = 8192,
        no_ref_audio: bool = False,
        edit_mask: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        r"""Sample zero-shot TTS latents with F5-TTS-style infilling.

        Args:
            data: Optional batch dict. If provided, `cond_latent`, `prompt`,
                `task`, `duration`, and `lens` are read from it when their
                explicit arguments are omitted.
            cond_latent: Reference speech latent, shaped `(B, T_ref, D)`.
            prompt: Transcript strings for the target speech.
            duration: Target duration in latent frames. May be an int shared
                by the batch or a `(B,)` tensor.
            lens: Number of valid reference latent frames per sample.

        Returns:
            Sampled latent `(B, T, D)`, optionally with the Euler trajectory.
        """

        self.eval()
        data = {} if data is None else dict(data)

        if cond_latent is None:
            cond_latent = data.get("cond_latent", data.get("reference_latent", data.get("input_latent")))
        if cond_latent is None:
            raise ValueError("ZeroShotTTSFlowMatcher.sample requires `cond_latent` or data['cond_latent'].")
        if cond_latent.dim() != 3:
            raise ValueError(f"`cond_latent` must have shape (B, T_ref, D), got {tuple(cond_latent.shape)}")

        cond_latent = cond_latent.to(self.device)
        dtype = cond_latent.dtype
        batch, cond_seq_len, latent_dim = cond_latent.shape
        device = cond_latent.device

        if prompt is None:
            prompt = data.get("prompt")
        if isinstance(prompt, str):
            prompt = [prompt]
        if prompt is None:
            raise ValueError("ZeroShotTTSFlowMatcher.sample requires `prompt` text.")
        if len(prompt) != batch:
            raise ValueError(f"Expected {batch} prompts, got {len(prompt)}.")

        if lens is None:
            lens = data.get("cond_length", data.get("input_length", data.get("target_length")))
        if lens is None:
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)
        else:
            lens = lens.to(device=device, dtype=torch.long)
        if torch.any(lens > cond_seq_len):
            raise ValueError("Every reference length in `lens` must be <= cond_latent.shape[1].")

        if duration is None:
            duration = data.get("duration", data.get("target_duration", data.get("target_length")))
        if duration is None:
            duration = lens + 1
        elif isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)
        else:
            duration = duration.to(device=device, dtype=torch.long)

        if steps <= 0:
            raise ValueError(f"`steps` must be positive, got {steps}.")
        if torch.any(lens >= max_duration):
            raise ValueError("Reference length must be smaller than `max_duration` so at least one frame can be generated.")

        duration = torch.maximum(duration, lens + 1).clamp(max=max_duration)
        sample_len = int(duration.max().item())
        if cond_seq_len > sample_len:
            cond_latent = cond_latent[:, :sample_len, :]
            cond_seq_len = sample_len

        cond_mask = _lens_to_mask(lens, length=cond_seq_len)
        if edit_mask is not None:
            edit_mask = edit_mask.to(device=device, dtype=torch.bool)
            if edit_mask.shape[-1] < cond_seq_len:
                edit_mask = F.pad(edit_mask, (0, cond_seq_len - edit_mask.shape[-1]), value=False)
            cond_mask = cond_mask & edit_mask[:, :cond_seq_len]

        cond = F.pad(cond_latent, (0, 0, 0, sample_len - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, sample_len - cond_mask.shape[-1]), value=False)
        if no_ref_audio:
            cond = torch.zeros_like(cond)
            cond_mask = torch.zeros_like(cond_mask)

        step_cond = torch.where(cond_mask[:, :, None], cond, torch.zeros_like(cond))
        target_mask = _lens_to_mask(duration, length=sample_len)

        sample_data = dict(data)
        task = data.get("task", ["text to speech"] * batch)
        if isinstance(task, str):
            task = [task] * batch
        sample_data["task"] = task
        sample_data["prompt"] = prompt
        sample_data["target_mask"] = target_mask
        sample_data["target_length"] = duration
        sample_data["cond_latent"] = step_cond
        sample_data["cond_mask"] = cond_mask

        cond_controls = self.adapter(
            sample_data,
            drop_audio_cond=no_ref_audio,
            drop_spk_cond=no_ref_audio,
            drop_text=False,
        )
        null_controls = None
        if abs(cfg_strength - 1.0) >= 1e-5:
            null_controls = self.adapter(
                sample_data,
                drop_audio_cond=True,
                drop_spk_cond=True,
                drop_text=True,
            )

        y0 = []
        for dur in duration.tolist():
            if seed is not None:
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, latent_dim, device=device, dtype=dtype))
        x = pad_sequence(y0, batch_first=True, padding_value=0.0)

        times = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
        if sway_sampling_coef is not None:
            times = times + sway_sampling_coef * (torch.cos(torch.pi / 2 * times) - 1 + times)

        trajectory = [x]
        for i in range(len(times) - 1):
            t = times[i]
            dt = times[i + 1] - t
            pred = self.base(t=t, x=x, controls=cond_controls)
            if null_controls is not None:
                null_pred = self.base(t=t, x=x, controls=null_controls)
                pred = null_pred + (pred - null_pred) * cfg_strength
            x = x + dt * pred
            x = torch.where(target_mask[:, :, None], x, torch.zeros_like(x))
            trajectory.append(x)

        out = torch.where(cond_mask[:, :, None], cond, x)
        if return_trajectory:
            return out, torch.stack(trajectory)
        return out

    def _random_span_mask(self, target_mask: Tensor) -> Tensor:
        lengths = target_mask.long().sum(dim=1)
        span_mask = torch.zeros_like(target_mask, dtype=torch.bool)

        min_frac, max_frac = self.frac_lengths_mask
        if not 0.0 < min_frac <= max_frac <= 1.0:
            raise ValueError(f"Invalid frac_lengths_mask: {self.frac_lengths_mask}")

        for i, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            if length <= 0:
                continue

            frac = torch.empty((), device=target_mask.device).uniform_(min_frac, max_frac).item()
            span_len = max(1, min(length, round(length * frac)))
            max_start = max(length - span_len, 0)
            start = int(torch.randint(0, max_start + 1, (), device=target_mask.device).item())
            span_mask[i, start : start + span_len] = True

        span_mask &= target_mask
        if not span_mask.any():
            raise RuntimeError("ZeroShotTTSFlowMatcher sampled an empty training span.")

        return span_mask


def _lens_to_mask(lens: Tensor, length: int | None = None) -> Tensor:
    if length is None:
        length = int(lens.max().item())
    return torch.arange(length, device=lens.device)[None, :] < lens[:, None]
