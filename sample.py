from __future__ import annotations

import argparse
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

import librosa
import soundfile
import torch
import torch.nn.functional as F
from torch import Tensor
import torchdiffeq
from torch.utils.data._utils.collate import default_collate
import numpy as np
import math
import torchaudio
from huggingface_hub import hf_hub_download

from audio_flow.utils import parse_yaml, to_device, load_vae, load_stereo
from audio_flow.solvers.euler import euler_solver
from train import get_model


SPEAKER_MODEL_REPO = "yfyeung/wavlm-large-speaker-verification"


def sample(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = args.config
    ckpt_path = args.ckpt_path
    out_path = args.out_path
    duration = args.duration
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Load model
    model = get_model(configs, ckpt_path).to(device)
    
    # Load VAE
    vae = load_vae(configs["train"].get("vae_type", "levo_vae")).to(device)

    with torch.no_grad():
        model.eval()
        ref_wav = args.ref_wav or args.input_path
        gen_txt = args.gen_txt or args.prompt
        ref_txt = args.ref_txt or args.ref_prompt
        if hasattr(model, "sample") and args.task == "text to speech" and ref_wav:
            if not gen_txt:
                raise ValueError("text to speech inference requires --gen_txt or --prompt.")
            ref_latent = compute_audio_vae(ref_wav, configs, duration=None, vae=vae)
            speaker_embedding = None
            if configs["adapter"]["name"] == "ZeroShotTTSSpeakerAdapter":
                if args.speaker_embedding_path:
                    speaker_embedding = load_speaker_embedding(args.speaker_embedding_path, torch.device(device))
                else:
                    speaker_model = load_speaker_embedding_model(torch.device(device))
                    speaker_embedding = compute_speaker_embedding(ref_wav, speaker_model, torch.device(device))

            gen_duration = get_tts_duration(gen_txt)
            gen_length = max(1, round(gen_duration * vae.fps))
            length = ref_latent.shape[0] + gen_length
            infer_prompt = gen_txt
            if ref_txt:
                infer_prompt = f"{ref_txt.strip()} {gen_txt.strip()}".strip()
            data = {
                "task": ["text to speech"],
                "prompt": [infer_prompt],
                "cond_latent": ref_latent[None, :, :],
                "cond_length": torch.tensor([ref_latent.shape[0]], device=device),
            }
            if speaker_embedding is not None:
                data["speaker_embedding"] = speaker_embedding[None, :]
            x_gen = model.sample(
                data=data,
                duration=length,
                steps=args.steps,
                cfg_strength=args.cfg_strength,
                sway_sampling_coef=args.sway_sampling_coef,
                seed=args.seed,
                max_duration=args.max_duration,
                no_ref_audio=args.no_ref_audio,
            )
            if args.full_out_path:
                audio_full = vae.decode(x_gen).data.cpu().numpy()[0]
                Path(args.full_out_path).parent.mkdir(parents=True, exist_ok=True)
                soundfile.write(file=args.full_out_path, data=audio_full.T, samplerate=vae.sr)
                print(f"Write full ref+gen audio to {args.full_out_path}")
            x_gen = x_gen[:, ref_latent.shape[0] :, :]
        else:
            duration = get_duration(args)
            length = round(duration * vae.fps)
            noise = torch.randn(1, length, vae.dim).to(device)
            data = get_data(args, duration, vae.fps)
            data = default_collate([data])
            data = to_device(data, device)

            controls = model.adapter(data)
            x_gen = euler_solver(model.base, noise, controls, n_steps=args.steps)  # (b, l, d)
    
    # Decode audio from VAE latents
    audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)

    # Write out
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
    print(f"Write out to {out_path}")


def get_data(args, duration, fps) -> dict:

    task = args.task
    configs = parse_yaml(args.config)

    if task in ["text to music", "text to speech", "text to audio"]:
        return {
            "task": task, 
            "prompt": args.prompt,
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    elif task in ["music source separation", "vocals to music", 
        "mono to stereo", "super-resolution", "codec to music"]:
        return {
            "task": task,
            "input_latent": compute_audio_vae(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    elif task in ["audio editing"]:
        return {
            "task": task,
            "prompt": args.prompt,
            "input_latent": compute_audio_vae(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    elif task in ["midi to audio"]:
        return {
            "task": task,
            "input_latent": compute_midi_roll(args.input_path, configs, duration),
            "target_mask": np.ones(int(duration * fps), dtype=bool)
        }

    else:
        raise ValueError(task)


def get_duration(args):
    task = args.task

    if task == "text to speech":
        return get_tts_duration(args.gen_txt or args.prompt)

    else:
        return args.duration


def get_tts_duration(text: str) -> float:
    return len(text) / 16.30


def compute_audio_vae(audio_path: str, configs: dict, duration: float | None, vae: torch.nn.Module | None = None) -> Tensor:

    # configs = parse_yaml(args.config)
    device = configs["train"]["device"]
    if vae is None:
        vae = load_vae(configs["train"].get("vae_type", "levo_vae")).to(device)

    sr = vae.sr
    audio = load_stereo(audio_path, sr)  # (c, l)
    if duration is not None:
        audio = librosa.util.fix_length(data=audio, size=int(duration * sr), axis=-1)
    audio = Tensor(audio).to(device)  # (c, l)
    latent = vae.encode(audio[None, :, :])[0]  # (t, d)
    return latent


def load_hf_ecapa_class():
    module_path = hf_hub_download(SPEAKER_MODEL_REPO, "ecapa_tdnn.py")
    spec = importlib.util.spec_from_file_location("hf_wavlm_ecapa_tdnn", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ECAPA_TDNN


def load_speaker_embedding_model(device: torch.device) -> torch.nn.Module:
    try:
        version("omegaconf")
    except PackageNotFoundError as exc:
        raise ModuleNotFoundError(
            "The WavLM speaker verification model loads its frontend through s3prl, "
            "which requires `omegaconf`. Install it in this environment with "
            "`python -m pip install omegaconf`, or pass --speaker_embedding_path "
            "to reuse a precomputed 256-d embedding."
        ) from exc

    ecapa_cls = load_hf_ecapa_class()
    model = ecapa_cls(
        feat_dim=1024,
        emb_dim=256,
        feat_type="wavlm_large",
        feature_selection="hidden_states",
        update_extract=False,
    )
    checkpoint_path = hf_hub_download(SPEAKER_MODEL_REPO, "wavlm-large.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected = set(incompatible.unexpected_keys)
    allowed_unexpected = {"loss_calculator.projection.weight"}
    if incompatible.missing_keys or unexpected - allowed_unexpected:
        raise RuntimeError(
            f"Unexpected speaker checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.requires_grad_(False).eval().to(device)
    return model


def load_speaker_embedding(path: str, device: torch.device) -> Tensor:
    embedding = torch.load(path, map_location="cpu")
    if isinstance(embedding, dict):
        for key in ("embedding", "speaker_embedding", "spk_embed", "xvector"):
            if key in embedding:
                embedding = embedding[key]
                break
        else:
            raise KeyError(
                f"Could not find a speaker embedding tensor in {path}. "
                f"Available keys: {sorted(embedding.keys())}"
            )
    embedding = torch.as_tensor(embedding, dtype=torch.float32, device=device).flatten()
    if embedding.shape[-1] != 256:
        raise RuntimeError(f"Expected speaker embedding dim 256, got {tuple(embedding.shape)} from {path}")
    if not torch.isfinite(embedding).all():
        raise FloatingPointError(f"Non-finite speaker embedding in {path}")
    return embedding


def load_audio_16k(audio_path: str, device: torch.device) -> Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0).float()
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, 16000).squeeze(0)
    return waveform.to(device)


@torch.inference_mode()
def compute_speaker_embedding(audio_path: str, model: torch.nn.Module, device: torch.device) -> Tensor:
    waveform = load_audio_16k(audio_path, device)
    embedding = model(waveform.unsqueeze(0)).squeeze(0).float()
    if embedding.shape[-1] != 256:
        raise RuntimeError(f"Expected speaker embedding dim 256, got {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise FloatingPointError(f"Non-finite speaker embedding for {audio_path}")
    return embedding


def compute_midi_roll(midi_path: str, configs: dict, duration: float) -> Tensor:

    from compute_latents.midi_io import read_single_track_midi

    fps = 100
    notes, pedals = read_single_track_midi(midi_path=midi_path, extend_pedal=True)

    midi_duration = max([note.end for note in notes])
    n_frames = math.ceil(midi_duration * fps)
    frame_roll = np.zeros((n_frames, 128), dtype=bool)
    onset_roll = np.zeros((n_frames, 128), dtype=bool)

    for note in notes:
        start = round(note.start * fps)
        end = round(note.end * fps)
        pitch = note.pitch
        velocity = note.velocity
        frame_roll[start : end, pitch] = True
        onset_roll[start, pitch] = True

    latent = np.concatenate([frame_roll, onset_roll], axis=-1)  # (l, d)
    latent = latent[0 : int(duration) * fps, :]
    return latent


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=False)
    parser.add_argument("--ref_prompt", type=str)
    parser.add_argument("--ref_wav", type=str)
    parser.add_argument("--ref_txt", type=str)
    parser.add_argument("--gen_txt", type=str)
    parser.add_argument("--speaker_embedding_path", type=str)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--full_out_path", type=str)

    parser.add_argument("--duration", type=float, default=10.)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg_strength", type=float, default=1.0)
    parser.add_argument("--sway_sampling_coef", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_duration", type=int, default=8192)
    parser.add_argument("--no_ref_audio", action="store_true", default=False)
    
    args = parser.parse_args()

    sample(args)
