from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import PartialState
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode LJSpeech with the ArchiTTS VAE.")
    parser.add_argument("--ljspeech-root", type=Path, default=Path("/datasets/jimmy/audio_flow_tts_full/LJSpeech-1.1"))
    parser.add_argument("--source-jsonl-root", type=Path, default=Path("/datasets/jimmy/audio_flow_tts_full/jsonls/tts"))
    parser.add_argument("--out-root", type=Path, default=Path("/datasets/jimmy/audio_flow_tts_architts_vae"))
    parser.add_argument("--vae-name", type=str, default="vae_24khz_f1920c64_1.0")
    parser.add_argument("--vae-ckpt-path", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--latent-mode", choices=["mean", "mode", "sample"], default="mean")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_architts_vae(vae_ckpt_path: Path | None, vae_name: str, device: torch.device, dtype: torch.dtype):
    from audio_flow.encoders.audio.architts_vae import ArchiTTSVAE

    wrapper = ArchiTTSVAE(ckpt_path=vae_ckpt_path, vae_name=vae_name).to(device)
    autoencoder = wrapper.model
    if dtype == torch.float16:
        autoencoder.model_half = True
    autoencoder.requires_grad_(False).eval()
    return autoencoder, wrapper.ckpt_path, wrapper.ckpt_path


def read_metadata(ljspeech_root: Path) -> dict[str, str]:
    metadata = {}
    with open(ljspeech_root / "metadata.csv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 3:
                metadata[parts[0]] = parts[2]
    return metadata


def id_from_source_latent(path: str) -> str:
    name = Path(path).stem
    return re.sub(r"_\d+_of_\d+$", "", name)


def read_split_ids(source_jsonl_root: Path, splits: list[str]) -> dict[str, list[str]]:
    split_ids = {}
    for split in splits:
        ids = []
        seen = set()
        path = source_jsonl_root / split / "ljspeech.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                meta = json.loads(line)
                audio_meta = meta["target"]["audio"]
                utt_id = id_from_source_latent(audio_meta["latent_path"])
                if utt_id not in seen:
                    seen.add(utt_id)
                    ids.append(utt_id)
        split_ids[split] = ids
    return split_ids


def valid_h5(path: Path, latent_mode: str) -> bool:
    if not path.exists():
        return False
    expected_mode = "mean" if latent_mode == "mode" else latent_mode
    try:
        with h5py.File(path, "r") as hf:
            mode = hf.attrs.get("latent_mode", None)
            return "latent" in hf and hf["latent"].ndim == 2 and hf["latent"].shape[1] == 64 and mode == expected_mode
    except OSError:
        return False


def encode_architts_latent(model, wav: torch.Tensor, latent_mode: str) -> torch.Tensor:
    if latent_mode == "sample":
        return model.encode(wav[None, :, :])[0]

    autoencoder = model.model
    if model.model_half:
        wav = wav.half()
        autoencoder.to(torch.float16)
    latent = autoencoder.encoder(wav[None, :, :]) if autoencoder.encoder is not None else wav[None, :, :]
    bottleneck = autoencoder.bottleneck
    if bottleneck is not None:
        bottleneck_name = type(bottleneck).__name__
        if bottleneck_name == "VAEBottleneck2":
            latent = latent[:, :-1, :]
        elif bottleneck_name == "VAEBottleneck":
            latent = latent.chunk(2, dim=1)[0]
        else:
            latent = bottleneck.encode(latent)
    if model.model_half:
        latent = latent.float()
    return (latent / model.scale)[0]


def encode_one(model, wav_path: Path, device: torch.device, dtype: torch.dtype, latent_mode: str) -> tuple[torch.Tensor, float]:
    wav, sr = torchaudio.load(wav_path)
    duration = wav.shape[-1] / sr
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != model.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    ratio = int(model.downsampling_ratio)
    remainder = wav.shape[-1] % ratio
    if remainder:
        wav = F.pad(wav, (0, ratio - remainder))
    wav = wav.to(device=device, dtype=dtype)
    with torch.inference_mode():
        latent = encode_architts_latent(model, wav, latent_mode).detach().float().cpu()
    # ArchiTTS VAE returns channel-first latent, audio_flow uses time-major latent.
    if latent.ndim != 2:
        raise RuntimeError(f"Expected 2-D latent, got {tuple(latent.shape)} for {wav_path}")
    if latent.shape[0] == 64:
        latent = latent.transpose(0, 1).contiguous()
    if latent.shape[1] != 64:
        raise RuntimeError(f"Expected latent dim 64, got {tuple(latent.shape)} for {wav_path}")
    if not torch.isfinite(latent).all():
        raise RuntimeError(f"Non-finite latent for {wav_path}")
    return latent, duration


def write_h5(path: Path, latent: torch.Tensor, duration: float, source_wav: Path, config_path: Path, ckpt_path: Path, latent_mode: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp_path, "w") as hf:
        hf.create_dataset("latent", data=latent.numpy(), compression="gzip", compression_opts=4)
        hf.attrs["source_wav"] = str(source_wav)
        hf.attrs["duration"] = float(duration)
        hf.attrs["latent_type"] = "architts_vae_24khz_f1920c64_1.0"
        hf.attrs["fps"] = 24000.0 / 1920.0
        hf.attrs["architts_config"] = str(config_path)
        hf.attrs["architts_checkpoint"] = str(ckpt_path)
        hf.attrs["latent_mode"] = "mean" if latent_mode == "mode" else latent_mode
    os.replace(tmp_path, path)


def write_jsonls(out_root: Path, split_ids: dict[str, list[str]], metadata: dict[str, str]) -> None:
    for split, ids in split_ids.items():
        out_path = out_root / "jsonls" / "tts" / split / "ljspeech.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for utt_id in ids:
                latent_path = out_root / "latents" / "ljspeech" / split / "audio" / f"{utt_id}.h5"
                with h5py.File(latent_path, "r") as hf:
                    duration = float(hf.attrs["duration"])
                    fps = float(hf.attrs["fps"])
                meta = {
                    "task": "text to speech",
                    "input": {"text": {"prompt": metadata[utt_id], "language": "en"}},
                    "target": {
                        "audio": {
                            "latent_path": str(latent_path),
                            "latent_type": "architts_vae_24khz_f1920c64_1.0",
                            "fps": fps,
                            "duration": duration,
                        }
                    },
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    state = PartialState()
    device = state.device
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32

    metadata = read_metadata(args.ljspeech_root)
    split_ids = read_split_ids(args.source_jsonl_root, args.splits)
    all_items = [(split, utt_id) for split, ids in split_ids.items() for utt_id in ids]
    if args.limit is not None:
        all_items = all_items[: args.limit]
    expected_items = set(all_items)

    model, config_path, ckpt_path = load_architts_vae(args.vae_ckpt_path, args.vae_name, device, dtype)

    with state.split_between_processes(all_items) as local_items:
        local_items = list(local_items)
        pbar = tqdm(local_items, desc=f"rank {state.process_index}", disable=not state.is_main_process)
        for split, utt_id in pbar:
            wav_path = args.ljspeech_root / "wavs" / f"{utt_id}.wav"
            h5_path = args.out_root / "latents" / "ljspeech" / split / "audio" / f"{utt_id}.h5"
            if not args.force and valid_h5(h5_path, args.latent_mode):
                continue
            latent, duration = encode_one(model, wav_path, device, dtype, args.latent_mode)
            write_h5(h5_path, latent, duration, wav_path, config_path, ckpt_path, args.latent_mode)

    state.wait_for_everyone()
    if state.is_main_process:
        missing = [
            (split, utt_id)
            for split, ids in split_ids.items()
            for utt_id in ids
            if (split, utt_id) in expected_items
            if not valid_h5(args.out_root / "latents" / "ljspeech" / split / "audio" / f"{utt_id}.h5", args.latent_mode)
        ]
        if missing:
            raise RuntimeError(f"Missing or invalid outputs: {missing[:10]} total={len(missing)}")
        if args.limit is None:
            write_jsonls(args.out_root, split_ids, metadata)
            for split, ids in split_ids.items():
                print(f"{split}: {len(ids)} files")
        else:
            print(f"Preview wrote {len(expected_items)} files")
        print(f"Wrote {args.out_root}")


if __name__ == "__main__":
    main()
