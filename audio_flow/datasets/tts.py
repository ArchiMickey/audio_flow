import h5py
import numpy as np
import torch

from audio_flow.datasets.ttm import TTMDataset


class FullLatentTTSDataset:
    r"""TTS dataset that loads full utterance latents without cropping.

    Unlike `TTSDataset`, this class ignores `clip_duration`. Each item returns
    the full latent sequence from its HDF5 file. Batching is handled by
    `full_latent_tts_collate`, which pads only to the longest utterance in the
    current batch and emits a per-sample `target_mask`.
    """

    def __init__(self, clip_duration: float | None = None):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        task = meta["task"]
        text_meta = meta["input"]["text"]
        prompt = text_meta["prompt"]
        speaker_id = text_meta.get("speaker_id")
        latent_path = meta["target"]["audio"]["latent_path"]

        with h5py.File(latent_path, "r") as hf:
            latent = hf["latent"][:].astype(np.float32)

        length = latent.shape[0]

        return {
            "task": task,
            "prompt": prompt,
            "target_latent": latent,
            "target_mask": np.ones(length, dtype=bool),
            "target_length": length,
            "target_latent_path": latent_path,
            "speaker_id": speaker_id,
        }


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

    return {
        "task": [sample["task"] for sample in samples],
        "prompt": [sample["prompt"] for sample in samples],
        "target_latent": torch.from_numpy(target_latent),
        "target_mask": torch.from_numpy(target_mask),
        "target_length": torch.from_numpy(target_length),
        "target_latent_path": [sample["target_latent_path"] for sample in samples],
        "speaker_id": [sample.get("speaker_id") for sample in samples],
    }


TTSDataset = TTMDataset
