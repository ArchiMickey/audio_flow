from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audio_flow.utils import load_vae, parse_yaml
from sample import (
    compute_audio_vae,
    compute_speaker_embedding,
    get_tts_duration,
    load_speaker_embedding,
    load_speaker_embedding_model,
)
from train import get_model


@dataclass
class TestItem:
    index: int
    ref_id: str
    ref_duration: float
    ref_text: str
    gen_id: str
    gen_duration: float
    gen_text: str


def parse_testset(path: Path) -> list[TestItem]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 6:
                raise ValueError(f"Expected 6 tab-separated fields at line {index + 1}, got {len(parts)}: {line}")
            ref_id, ref_duration, ref_text, gen_id, gen_duration, gen_text = parts
            items.append(
                TestItem(
                    index=index,
                    ref_id=ref_id,
                    ref_duration=float(ref_duration),
                    ref_text=ref_text,
                    gen_id=gen_id,
                    gen_duration=float(gen_duration),
                    gen_text=gen_text,
                )
            )
    return items


def librispeech_audio_path(root: Path, utterance_id: str) -> Path:
    speaker, chapter, _ = utterance_id.split("-", 2)
    for subset in ("test-clean", "dev-clean", "test-other", "dev-other", "train-clean-100", "train-clean-360", "train-other-500"):
        path = root / subset / speaker / chapter / f"{utterance_id}.flac"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find LibriSpeech audio for {utterance_id} under {root}")


def speaker_embedding_path(root: Path, utterance_id: str) -> Path:
    speaker, chapter, _ = utterance_id.split("-", 2)
    return root / "test-clean" / speaker / chapter / f"{utterance_id}.pt"


def write_jsonl_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--testset", default="testset/librispeech_pc_test_clean_cross_sentence.lst")
    parser.add_argument("--librispeech_root", default="/datasets/LibriSpeech")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--speaker_embedding_root", default="/datasets/jimmy/projects/architts/ecapa_libritts")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg_strength", type=float, default=2.0)
    parser.add_argument("--sway_sampling_coef", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_duration", type=int, default=8192)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_full", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    full_wav_dir = out_dir / "full_wav"
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    if args.save_full:
        full_wav_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    configs = parse_yaml(args.config)
    device = torch.device(configs["train"]["device"])
    model = get_model(configs, args.ckpt_path).to(device).eval()
    vae = load_vae(configs["train"].get("vae_type", "levo_vae")).to(device).eval()

    speaker_model = None
    items = parse_testset(Path(args.testset))
    if args.limit is not None:
        items = items[: args.limit]

    records = []
    for item in tqdm(items, desc="infer"):
        out_path = wav_dir / f"{item.index:04d}_{item.ref_id}_to_{item.gen_id}.wav"
        full_out_path = full_wav_dir / out_path.name if args.save_full else None
        if out_path.exists() and not args.overwrite:
            records.append({"index": item.index, "out_path": str(out_path), "skipped_existing": True})
            continue

        ref_wav = librispeech_audio_path(Path(args.librispeech_root), item.ref_id)
        ref_latent = compute_audio_vae(str(ref_wav), configs, duration=None, vae=vae)

        spk_path = speaker_embedding_path(Path(args.speaker_embedding_root), item.ref_id)
        if spk_path.exists():
            speaker_embedding = load_speaker_embedding(str(spk_path), device)
        else:
            if speaker_model is None:
                speaker_model = load_speaker_embedding_model(device)
            speaker_embedding = compute_speaker_embedding(str(ref_wav), speaker_model, device)

        gen_length = max(1, round(get_tts_duration(item.gen_text) * vae.fps))
        length = ref_latent.shape[0] + gen_length
        infer_prompt = f"{item.ref_text.strip()} {item.gen_text.strip()}".strip()
        data = {
            "task": ["text to speech"],
            "prompt": [infer_prompt],
            "cond_latent": ref_latent[None, :, :],
            "cond_length": torch.tensor([ref_latent.shape[0]], device=device),
            "speaker_embedding": speaker_embedding[None, :],
        }

        x_gen = model.sample(
            data=data,
            duration=length,
            steps=args.steps,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            seed=args.seed,
            max_duration=args.max_duration,
        )
        if full_out_path is not None:
            audio_full = vae.decode(x_gen).data.cpu().numpy()[0]
            soundfile.write(full_out_path, audio_full.T, samplerate=vae.sr)
        gen_latent = x_gen[:, ref_latent.shape[0] :, :]
        audio_gen = vae.decode(gen_latent).data.cpu().numpy()[0]
        soundfile.write(out_path, audio_gen.T, samplerate=vae.sr)

        record = {
            **asdict(item),
            "ref_wav": str(ref_wav),
            "speaker_embedding_path": str(spk_path) if spk_path.exists() else None,
            "out_path": str(out_path),
            "full_out_path": str(full_out_path) if full_out_path is not None else None,
            "sample_rate": vae.sr,
            "steps": args.steps,
            "cfg_strength": args.cfg_strength,
            "seed": args.seed,
            "checkpoint": args.ckpt_path,
            "config": args.config,
        }
        records.append(record)
        write_jsonl_record(manifest_path, record)

    summary = {
        "num_items": len(items),
        "num_records": len(records),
        "config": args.config,
        "checkpoint": args.ckpt_path,
        "testset": args.testset,
        "librispeech_root": args.librispeech_root,
        "out_dir": str(out_dir),
        "steps": args.steps,
        "cfg_strength": args.cfg_strength,
        "seed": args.seed,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} records to {out_dir}")


if __name__ == "__main__":
    main()
