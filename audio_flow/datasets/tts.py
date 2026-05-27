import h5py
import numpy as np
import torch
from pathlib import Path

from audio_flow.datasets.ttm import TTMDataset


class FullLatentTTSDataset:
    r"""TTS dataset that loads full utterance latents without cropping.

    Unlike `TTSDataset`, this class ignores `clip_duration`. Each item returns
    the full latent sequence from its HDF5 file. Batching is handled by
    `full_latent_tts_collate`, which pads only to the longest utterance in the
    current batch and emits a per-sample `target_mask`.
    """

    def __init__(
        self,
        clip_duration: float | None = None,
        speaker_embedding_root: str | None = None,
        require_speaker_embedding: bool = False,
    ):
        self.clip_duration = clip_duration
        self.speaker_embedding_root = Path(speaker_embedding_root) if speaker_embedding_root else None
        self.require_speaker_embedding = require_speaker_embedding

    def __getitem__(self, meta: dict) -> dict:
        task = meta["task"]
        text_meta = meta["input"]["text"]
        prompt = text_meta["prompt"]
        speaker_id = text_meta.get("speaker_id")
        latent_path = meta["target"]["audio"]["latent_path"]

        with h5py.File(latent_path, "r") as hf:
            latent = hf["latent"][:].astype(np.float32)

        length = latent.shape[0]

        sample = {
            "task": task,
            "prompt": prompt,
            "target_latent": latent,
            "target_mask": np.ones(length, dtype=bool),
            "target_length": length,
            "target_latent_path": latent_path,
            "speaker_id": speaker_id,
        }

        if self.speaker_embedding_root is not None:
            speaker_embedding_path = self._speaker_embedding_path(latent_path)
            if not speaker_embedding_path.exists():
                if self.require_speaker_embedding:
                    raise FileNotFoundError(f"Missing speaker embedding: {speaker_embedding_path}")
            else:
                speaker_embedding = torch.load(speaker_embedding_path, map_location="cpu")
                if isinstance(speaker_embedding, dict):
                    for key in ("embedding", "speaker_embedding", "spk_embed", "xvector"):
                        if key in speaker_embedding:
                            speaker_embedding = speaker_embedding[key]
                            break
                    else:
                        raise KeyError(
                            f"Could not find a speaker embedding tensor in {speaker_embedding_path}. "
                            f"Available keys: {sorted(speaker_embedding.keys())}"
                        )
                speaker_embedding = torch.as_tensor(speaker_embedding, dtype=torch.float32).flatten()
                sample["speaker_embedding"] = speaker_embedding.numpy()
                sample["speaker_embedding_path"] = str(speaker_embedding_path)

        return sample

    def _speaker_embedding_path(self, latent_path: str) -> Path:
        stem = Path(latent_path).stem
        parts = stem.split("_")
        if len(parts) < 2:
            raise ValueError(f"Cannot infer LibriTTS speaker embedding path from latent stem: {stem}")
        speaker_id, chapter_id = parts[0], parts[1]

        subset = None
        for part in Path(latent_path).parts:
            if part.startswith(("train-", "dev-", "test-")):
                subset = part
        if subset is None:
            raise ValueError(f"Cannot infer LibriTTS subset from latent path: {latent_path}")

        return self.speaker_embedding_root / subset / speaker_id / chapter_id / f"{stem}.pt"


def full_latent_tts_collate(samples: list[dict]) -> dict:
    r"""Pad full-length TTS latents to the longest sample in the batch."""

    max_length = max(sample["target_latent"].shape[0] for sample in samples)
    latent_dim = samples[0]["target_latent"].shape[1]
    batch_size = len(samples)

    target_latent = np.zeros((batch_size, max_length, latent_dim), dtype=np.float32)
    target_mask = np.zeros((batch_size, max_length), dtype=bool)
    target_length = np.zeros((batch_size,), dtype=np.int64)

    for i, sample in enumerate(samples):
        latent = sample["target_latent"]
        length = latent.shape[0]
        target_latent[i, :length] = latent
        target_mask[i, :length] = True
        target_length[i] = length

    batch = {
        "task": [sample["task"] for sample in samples],
        "prompt": [sample["prompt"] for sample in samples],
        "target_latent": torch.from_numpy(target_latent),
        "target_mask": torch.from_numpy(target_mask),
        "target_length": torch.from_numpy(target_length),
        "target_latent_path": [sample["target_latent_path"] for sample in samples],
        "speaker_id": [sample.get("speaker_id") for sample in samples],
    }

    has_speaker_embedding = ["speaker_embedding" in sample for sample in samples]
    if any(has_speaker_embedding):
        if not all(has_speaker_embedding):
            raise ValueError("Mixed batch with and without speaker_embedding is not supported.")
        speaker_embeddings = np.stack([sample["speaker_embedding"] for sample in samples]).astype(np.float32)
        batch["speaker_embedding"] = torch.from_numpy(speaker_embeddings)
        batch["speaker_embedding_path"] = [sample["speaker_embedding_path"] for sample in samples]

    return batch


TTSDataset = TTMDataset
