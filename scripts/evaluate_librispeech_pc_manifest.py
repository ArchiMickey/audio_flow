from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAVLM_CKPT = REPO_ROOT / "checkpoints/ecapa_tdnn/wavlm_large_finetune.pth"


def parse_gpu_ids(value: str) -> list[int]:
    ids = []
    for item in value.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    if not ids:
        raise ValueError("--gpu_ids must contain at least one GPU id")
    return ids


def split_jobs(test_set: list[tuple[str, str, str]], gpu_ids: list[int]) -> list[tuple[int, list[tuple[str, str, str]]]]:
    if len(gpu_ids) == 1:
        return [(gpu_ids[0], test_set)]

    wav_per_job = len(test_set) // len(gpu_ids) + 1
    return [
        (gpu_id, test_set[i * wav_per_job : (i + 1) * wav_per_job])
        for i, gpu_id in enumerate(gpu_ids)
    ]


def load_manifest_test_set(manifest_path: Path, *, limit: int | None = None) -> list[tuple[str, str, str]]:
    test_set = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            gen_wav = Path(record["out_path"])
            ref_wav = Path(record["ref_wav"])
            if not gen_wav.exists():
                raise FileNotFoundError(f"Generated wav not found: {gen_wav}")
            if not ref_wav.exists():
                raise FileNotFoundError(f"Reference wav not found: {ref_wav}")
            test_set.append((str(gen_wav), str(ref_wav), record["gen_text"]))
            if limit is not None and len(test_set) >= limit:
                break
    if not test_set:
        raise RuntimeError(f"No evaluation rows loaded from {manifest_path}")
    return test_set


def write_results(result_path: Path, rows: list[dict], metric_key: str) -> float:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = []
    with result_path.open("w", encoding="utf-8") as f:
        for row in rows:
            value = row[metric_key]
            metrics.append(value)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        metric = float(np.mean(metrics))
        f.write(f"\n{metric_key.upper()}: {metric:.5f}\n")
    return metric


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate audio_flow LibriSpeech-PC manifest outputs with ArchiTTS eval metrics."
    )
    parser.add_argument("--manifest", required=True, help="Inference manifest.jsonl produced by audio_flow.")
    parser.add_argument(
        "--eval_task",
        default="all",
        choices=["wer", "sim", "ssim", "all"],
        help="Metric to run. `ssim` is kept as an alias for ArchiTTS speaker similarity.",
    )
    parser.add_argument("--lang", default="en", choices=["en", "zh"])
    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument("--out_dir", help="Directory for metric JSONL and summary files.")
    parser.add_argument("--asr_ckpt_path", default="")
    parser.add_argument(
        "--wavlm_ckpt_path",
        default=str(DEFAULT_WAVLM_CKPT),
    )
    parser.add_argument("--limit", type=int, help="Evaluate only the first N manifest rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing metric result files.")
    return parser.parse_args()


def main() -> None:
    args = get_args()
    from audio_flow.eval.architts_metrics import run_asr_wer, run_sim

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent / "eval"
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    test_set = load_manifest_test_set(manifest_path, limit=args.limit)
    jobs = split_jobs(test_set, gpu_ids)

    task_names = ["wer", "ssim"] if args.eval_task == "all" else [args.eval_task]
    summary = {
        "manifest": str(manifest_path),
        "num_samples": len(test_set),
        "gpu_ids": gpu_ids,
        "metrics": {},
    }

    for task_name in task_names:
        metric_key = "sim" if task_name == "sim" else task_name
        result_path = out_dir / f"_{task_name}_results.jsonl"
        if result_path.exists() and not args.force:
            raise FileExistsError(f"{result_path} already exists. Pass --force to overwrite.")

        full_results: list[dict] = []
        if task_name == "wer":
            with mp.Pool(len(gpu_ids)) as pool:
                pool_args = [(rank, args.lang, sub_test_set, args.asr_ckpt_path) for rank, sub_test_set in jobs]
                for result in pool.map(run_asr_wer, pool_args):
                    full_results.extend(result)
        elif task_name in {"sim", "ssim"}:
            with mp.Pool(len(gpu_ids)) as pool:
                pool_args = [(rank, sub_test_set, args.wavlm_ckpt_path) for rank, sub_test_set in jobs]
                for result in pool.map(run_sim, pool_args):
                    full_results.extend(result)
            if task_name == "ssim":
                for row in full_results:
                    row["ssim"] = row.pop("sim")
        else:
            raise ValueError(task_name)

        metric = write_results(result_path, full_results, metric_key)
        summary["metrics"][task_name] = {
            "mean": round(metric, 5),
            "result_path": str(result_path),
            "num_samples": len(full_results),
        }
        print(f"{task_name.upper()} {metric:.5f} over {len(full_results)} samples")
        print(f"Saved {task_name} results to {result_path}")

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
