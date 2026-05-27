from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F
import torchaudio
from accelerate import PartialState
from tqdm import tqdm


@dataclass(frozen=True)
class LibriTTSItem:
    split: str
    subset: str
    wav_path: Path
    text: str

    @property
    def utt_id(self) -> str:
        return self.wav_path.stem

    @property
    def rel_parent(self) -> Path:
        return Path(*self.wav_path.parts[-3:-1])

    @property
    def speaker_id(self) -> str:
        return self.wav_path.parts[-3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode LibriTTS with the ArchiTTS VAE.")
    parser.add_argument("--libritts-root", type=Path, default=Path("/datasets/LibriTTS"))
    parser.add_argument("--out-root", type=Path, default=Path("/datasets/jimmy/audio_flow_tts_libritts_architts_vae_mean"))
    parser.add_argument("--vae-name", type=str, default="vae_24khz_f1920c64_1.0")
    parser.add_argument("--vae-ckpt-path", type=Path)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train-clean-100", "train-clean-360", "train-other-500", "test-clean"],
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--latent-mode", choices=["mean", "mode", "sample"], default="mean")
    parser.add_argument("--no-wait-for-everyone", action="store_true")
    parser.add_argument("--skip-jsonls", action="store_true")
    parser.add_argument("--write-jsonls-only", action="store_true")
    parser.add_argument("--item-shard-index", type=int, default=0)
    parser.add_argument("--item-num-shards", type=int, default=1)
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


def split_from_subset(subset: str) -> str:
    if subset.startswith("test"):
        return "test"
    if subset.startswith(("dev", "valid")):
        return "valid"
    return "train"


def discover_items(libritts_root: Path, subsets: list[str], limit: int | None) -> list[LibriTTSItem]:
    items: list[LibriTTSItem] = []
    for subset in subsets:
        subset_dir = libritts_root / subset
        for wav_path in sorted(subset_dir.rglob("*.wav")):
            text_path = wav_path.with_suffix(".normalized.txt")
            if not text_path.exists():
                continue
            text = text_path.read_text(encoding="utf-8").strip()
            items.append(LibriTTSItem(split=split_from_subset(subset), subset=subset, wav_path=wav_path, text=text))
            if limit is not None and len(items) >= limit:
                return items
    return items


def h5_path(out_root: Path, item: LibriTTSItem) -> Path:
    return out_root / "latents" / "libritts" / item.split / item.subset / "audio" / item.rel_parent / f"{item.utt_id}.h5"


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
    if latent.ndim != 2:
        raise RuntimeError(f"Expected 2-D latent, got {tuple(latent.shape)} for {wav_path}")
    if latent.shape[0] == 64:
        latent = latent.transpose(0, 1).contiguous()
    if latent.shape[1] != 64:
        raise RuntimeError(f"Expected latent dim 64, got {tuple(latent.shape)} for {wav_path}")
    if not torch.isfinite(latent).all():
        raise RuntimeError(f"Non-finite latent for {wav_path}")
    return latent, duration


def write_h5(
    path: Path,
    latent: torch.Tensor,
    duration: float,
    item: LibriTTSItem,
    config_path: Path,
    ckpt_path: Path,
    latent_mode: str,
) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(tmp_path, "w") as hf:
        hf.create_dataset("latent", data=latent.numpy(), compression="gzip", compression_opts=4)
        hf.attrs["source_wav"] = str(item.wav_path)
        hf.attrs["duration"] = float(duration)
        hf.attrs["latent_type"] = "architts_vae_24khz_f1920c64_1.0"
        hf.attrs["fps"] = 24000.0 / 1920.0
        hf.attrs["architts_config"] = str(config_path)
        hf.attrs["architts_checkpoint"] = str(ckpt_path)
        hf.attrs["latent_mode"] = "mean" if latent_mode == "mode" else latent_mode
        hf.attrs["libritts_subset"] = item.subset
    os.replace(tmp_path, path)


def write_jsonls(out_root: Path, items: list[LibriTTSItem], latent_mode: str) -> None:
    handles = {}
    counts = {}

    def get_handle(path: Path):
        if path not in handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            handles[path] = open(path, "w", encoding="utf-8")
            counts[path] = 0
        return handles[path]

    try:
        for item in items:
            latent_path = h5_path(out_root, item)
            with h5py.File(latent_path, "r") as hf:
                latent_len = int(hf["latent"].shape[0])
                duration = float(hf.attrs["duration"])
                fps = float(hf.attrs["fps"])
            meta = {
                "task": "text to speech",
                "input": {"text": {"prompt": item.text, "language": "en", "speaker_id": item.speaker_id}},
                "target": {
                    "audio": {
                        "latent_path": str(latent_path),
                        "latent_type": "architts_vae_24khz_f1920c64_1.0",
                        "fps": fps,
                        "duration": duration,
                        "latent_length": latent_len,
                    }
                },
            }
            line = json.dumps(meta, ensure_ascii=False) + "\n"
            for path in [
                out_root / "jsonls" / "tts" / item.split / f"libritts_{item.subset}.jsonl",
                out_root / "jsonls" / "tts" / item.split / "libritts.jsonl",
            ]:
                get_handle(path).write(line)
                counts[path] += 1
    finally:
        for handle in handles.values():
            handle.close()

    by_split = {
        path: count
        for path, count in counts.items()
        if path.name == "libritts.jsonl"
    }
    for path, count in sorted(by_split.items()):
        print(f"{path}: {count} files")


def main() -> None:
    args = parse_args()
    if args.write_jsonls_only:
        items = discover_items(args.libritts_root, args.subsets, args.limit)
        write_jsonls(args.out_root, items, args.latent_mode)
        print(f"Wrote {args.out_root}")
        return

    state = PartialState()
    device = state.device
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32
    items = discover_items(args.libritts_root, args.subsets, args.limit)
    if args.item_num_shards < 1:
        raise ValueError(f"--item-num-shards must be >= 1, got {args.item_num_shards}")
    if not 0 <= args.item_shard_index < args.item_num_shards:
        raise ValueError(
            f"--item-shard-index must be in [0, {args.item_num_shards}), got {args.item_shard_index}"
        )
    items = items[args.item_shard_index :: args.item_num_shards]
    model, config_path, ckpt_path = load_architts_vae(args.vae_ckpt_path, args.vae_name, device, dtype)

    with state.split_between_processes(items) as local_items:
        local_items = list(local_items)
        pbar = tqdm(local_items, desc=f"rank {state.process_index}", disable=not state.is_main_process)
        for item in pbar:
            out_path = h5_path(args.out_root, item)
            if not args.force and valid_h5(out_path, args.latent_mode):
                continue
            latent, duration = encode_one(model, item.wav_path, device, dtype, args.latent_mode)
            write_h5(out_path, latent, duration, item, config_path, ckpt_path, args.latent_mode)

    if not args.no_wait_for_everyone:
        state.wait_for_everyone()
    if state.is_main_process:
        if args.skip_jsonls:
            print(f"Extraction shard finished under {args.out_root}; skip jsonl writing")
            return
        missing = [item for item in items if not valid_h5(h5_path(args.out_root, item), args.latent_mode)]
        if missing:
            raise RuntimeError(f"Missing or invalid outputs: {missing[:10]} total={len(missing)}")
        write_jsonls(args.out_root, items, args.latent_mode)
        print(f"Wrote {args.out_root}")


if __name__ == "__main__":
    main()
