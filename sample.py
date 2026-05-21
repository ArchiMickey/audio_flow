from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import soundfile
import torch
from torch import Tensor
import torchdiffeq
from torch.utils.data._utils.collate import default_collate
import numpy as np
import math

from audio_flow.utils import parse_yaml, to_device, load_vae, load_stereo
from audio_flow.solvers.euler import euler_solver
from train import get_model


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
        if hasattr(model, "sample") and args.task == "text to speech" and args.input_path:
            ref_latent = compute_audio_vae(args.input_path, configs, duration=None)
            gen_duration = get_duration(args)
            gen_length = max(1, round(gen_duration * vae.fps))
            length = ref_latent.shape[0] + gen_length
            infer_prompt = args.prompt
            if args.ref_prompt:
                infer_prompt = f"{args.ref_prompt.strip()} {args.prompt.strip()}".strip()
            data = {
                "task": ["text to speech"],
                "prompt": [infer_prompt],
                "cond_latent": ref_latent[None, :, :],
                "cond_length": torch.tensor([ref_latent.shape[0]], device=device),
            }
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
        return len(args.prompt) / 16.30

    else:
        return args.duration


def compute_audio_vae(audio_path: str, configs: dict, duration: float | None) -> Tensor:

    # configs = parse_yaml(args.config)
    device = configs["train"]["device"]
    vae = load_vae(configs["train"].get("vae_type", "levo_vae")).to(device)

    sr = vae.sr
    audio = load_stereo(audio_path, sr)  # (c, l)
    if duration is not None:
        audio = librosa.util.fix_length(data=audio, size=int(duration * sr), axis=-1)
    audio = Tensor(audio).to(device)  # (c, l)
    latent = vae.encode(audio[None, :, :])[0]  # (t, d)
    return latent


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
    parser.add_argument("--out_path", type=str, required=True)

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
