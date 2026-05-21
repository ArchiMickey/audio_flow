import random

import h5py

from audio_flow.utils import load_jsonl


class BatchJsonlSampler:
    r"""Jsonl Sampler."""
    def __init__(self, jsonl_paths: list[str], weights: list[float], batch_size: int):
        self.jsonl_paths = jsonl_paths
        self.weights = weights
        self.batch_size = batch_size

        self.metas = [load_jsonl(path) for path in self.jsonl_paths]  # list[list[str]]
        self.lens = [len(metas) for metas in self.metas]  # list[str]
        self.ptrs = [0 for _ in self.lens]  # list[int]
        self.indices = [self.random_indices(L) for L in self.lens]  # list[list[int]]

    def __iter__(self) -> dict:
        r"""Random sample a jsonl file and sample a line."""

        while True:

            # Randomly sample a jsonl file
            i = random.choices(population=range(len(self.metas)), weights=self.weights, k=1)[0]
            batch_meta = []
            
            for _ in range(self.batch_size):
                # Reset pointer and shuffle indices
                if self.ptrs[i] == len(self.metas[i]):
                    self.indices[i] = self.random_indices(self.lens[i])
                    self.ptrs[i] = 0

                # Randomly sample an item in the jsonl file
                j = self.indices[i][self.ptrs[i]]  # item index
                self.ptrs[i] += 1

                meta = self.metas[i][j]
                batch_meta.append(meta)

            yield batch_meta

    def random_indices(self, N: int) -> list[int]:
        indices = list(range(N))
        random.shuffle(indices)
        return indices


class StochasticDynamicBatchJsonlSampler:
    r"""Epoch-shuffled dynamic batch sampler for JSONL metadata.

    Batches are capped by the sum of real latent frames, not by the padded dense
    tensor size after collation. Each epoch shuffles examples before greedy
    batching, then shuffles the produced batches.
    """

    def __init__(
        self,
        jsonl_paths: list[str],
        weights: list[float] | None = None,
        max_tokens_per_batch: int = 2048,
        max_examples_per_batch: int | None = None,
        drop_last: bool = False,
        length_source: str = "metadata",
        seed: int | None = None,
    ):
        if max_tokens_per_batch <= 0:
            raise ValueError(f"`max_tokens_per_batch` must be positive, got {max_tokens_per_batch}.")
        if max_examples_per_batch is not None and max_examples_per_batch <= 0:
            raise ValueError(f"`max_examples_per_batch` must be positive, got {max_examples_per_batch}.")
        if length_source not in {"metadata", "h5"}:
            raise ValueError(f"`length_source` must be 'metadata' or 'h5', got {length_source}.")

        self.jsonl_paths = jsonl_paths
        self.weights = weights or [1.0 for _ in jsonl_paths]
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_examples_per_batch = max_examples_per_batch
        self.drop_last = drop_last
        self.length_source = length_source
        self.seed = seed
        self.epoch = 0

        self.metas = [load_jsonl(path) for path in self.jsonl_paths]
        self.lengths = [[self.get_length(meta) for meta in metas] for metas in self.metas]

    def __iter__(self):
        while True:
            rng = random.Random(None if self.seed is None else self.seed + self.epoch)
            epoch_batches = [
                self.build_epoch_batches(metas, lengths, rng)
                for metas, lengths in zip(self.metas, self.lengths)
            ]

            while any(epoch_batches):
                available = [i for i, batches in enumerate(epoch_batches) if batches]
                weights = [self.weights[i] for i in available]
                source_id = rng.choices(available, weights=weights, k=1)[0]
                yield epoch_batches[source_id].pop()

            self.epoch += 1

    def build_epoch_batches(
        self,
        metas: list[dict],
        lengths: list[int],
        rng: random.Random,
    ) -> list[list[dict]]:
        indices = list(range(len(metas)))
        rng.shuffle(indices)

        batches = []
        batch = []
        num_tokens = 0

        for idx in indices:
            length = lengths[idx]
            exceeds_tokens = batch and num_tokens + length > self.max_tokens_per_batch
            exceeds_examples = (
                self.max_examples_per_batch is not None
                and len(batch) >= self.max_examples_per_batch
            )

            if exceeds_tokens or exceeds_examples:
                batches.append(batch)
                batch = []
                num_tokens = 0

            batch.append(metas[idx])
            num_tokens += length

        if batch and not self.drop_last:
            batches.append(batch)

        rng.shuffle(batches)
        return batches

    def get_length(self, meta: dict) -> int:
        audio_meta = meta["target"]["audio"]

        if self.length_source == "h5":
            with h5py.File(audio_meta["latent_path"], "r") as hf:
                return int(hf["latent"].shape[0])

        for key in ["latent_length", "num_frames", "length"]:
            if key in audio_meta:
                return int(audio_meta[key])

        if "target_length" in meta:
            return int(meta["target_length"])

        if "duration" in audio_meta and "fps" in audio_meta:
            return max(1, round(float(audio_meta["duration"]) * float(audio_meta["fps"])))

        with h5py.File(audio_meta["latent_path"], "r") as hf:
            return int(hf["latent"].shape[0])
