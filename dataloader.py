import queue
import random
import threading
import zlib
from dataclasses import dataclass
from functools import lru_cache

import tiktoken
import torch
from datasets import load_dataset as hf_load_dataset
from huggingface_hub import HfFileSystem

DATASET_SHUFFLE_SEED = 42
DATASET_SHUFFLE_BUFFER_SIZE = 10_000
MAX_ZLIB_COMPRESSION_RATIO = 3.0
VALIDATION_SIZE_FRACTION = 0.01
VALIDATION_SPLIT_SEED = 42
HF_FS = HfFileSystem()


@dataclass(frozen=True)
class DatasetSpec:
    namespace: str
    glob_pattern: str
    examples_per_batch: int

    @property
    def source(self) -> str:
        return f"{self.namespace}:{self.glob_pattern}"


@dataclass(frozen=True)
class DatasetStream:
    source: str
    dataset: object
    examples_per_batch: int


DATASETS = [
    DatasetSpec("HuggingFaceFW/fineweb-edu", "sample/10BT/**/*.parquet", 4),
    DatasetSpec("mlfoundations/dclm-baseline-1.0", "**/*.zst", 4),
    DatasetSpec("HuggingFaceTB/finemath", "finemath-3plus/**/*.parquet", 1),
    DatasetSpec("HuggingFaceTB/finemath", "infiwebmath-3plus/**/*.parquet", 1),
]


def _extract_text(sample):
    if isinstance(sample, str):
        return sample
    if isinstance(sample, dict) and isinstance(sample.get("text"), str):
        return sample["text"]
    raise TypeError(
        "dataset samples must be strings or dicts with a string 'text' field"
    )


def _compression_ratio(text):
    text_bytes = text.encode("utf-8")
    if not text_bytes:
        return 0.0
    compressed = zlib.compress(text_bytes)
    return len(text_bytes) / max(len(compressed), 1)


def _passes_zlib_filter(sample):
    return _compression_ratio(_extract_text(sample)) <= MAX_ZLIB_COMPRESSION_RATIO


@lru_cache(maxsize=None)
def _glob_file_info(namespace, glob_pattern):
    repo_prefix = f"datasets/{namespace}/"
    matches = sorted(HF_FS.glob(f"{repo_prefix}{glob_pattern}"))
    files = []
    for match in matches:
        info = HF_FS.info(match)
        if info.get("type") != "file":
            continue
        files.append((match.removeprefix(repo_prefix), int(info["size"])))
    if not files:
        raise ValueError(f"No files matched {namespace}:{glob_pattern}")
    return tuple(files)


@lru_cache(maxsize=None)
def _split_data_files(namespace, glob_pattern, validation_fraction, seed):
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0.0, 1.0)")

    files = list(_glob_file_info(namespace, glob_pattern))
    all_paths = [path for path, _ in files]
    if len(files) == 1 or validation_fraction == 0.0:
        return tuple(all_paths), tuple()

    total_size = sum(size for _, size in files)
    target_validation_size = total_size * validation_fraction
    rng = random.Random(f"{seed}:{namespace}:{glob_pattern}")
    shuffled_files = files.copy()
    rng.shuffle(shuffled_files)

    validation_files = []
    validation_size = 0
    for file_idx, (path, size) in enumerate(shuffled_files):
        remaining_files = len(shuffled_files) - file_idx - 1
        if validation_size >= target_validation_size or remaining_files == 0:
            break
        validation_files.append(path)
        validation_size += size

    validation_set = set(validation_files)
    train_files = [path for path in all_paths if path not in validation_set]
    val_files = [path for path in all_paths if path in validation_set]
    if not train_files:
        raise ValueError(
            f"Validation split consumed every file for {namespace}:{glob_pattern}"
        )
    return tuple(train_files), tuple(val_files)


def _build_stream(spec, data_files, *, num_shards=1, shard_index=0, skip_examples=0):
    if not data_files:
        raise ValueError(f"No files selected for {spec.source}")
    split = next(iter(data_files))

    dataset = hf_load_dataset(
        spec.namespace,
        split=split,
        streaming=True,
        data_files={split: list(data_files[split])},
    )
    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
    dataset = dataset.filter(_passes_zlib_filter)
    dataset = dataset.shuffle(
        seed=DATASET_SHUFFLE_SEED, buffer_size=DATASET_SHUFFLE_BUFFER_SIZE
    )
    if num_shards > 1:
        dataset = dataset.shard(num_shards=num_shards, index=shard_index)
    if skip_examples > 0:
        dataset = dataset.skip(skip_examples)
    return DatasetStream(
        source=spec.source,
        dataset=dataset,
        examples_per_batch=spec.examples_per_batch,
    )


def _load_stream(spec, *, num_shards=1, shard_index=0, skip_examples=0):
    train_files, validation_files = _split_data_files(
        spec.namespace,
        spec.glob_pattern,
        VALIDATION_SIZE_FRACTION,
        VALIDATION_SPLIT_SEED,
    )
    train_stream = _build_stream(
        spec,
        {"train": train_files},
        num_shards=num_shards,
        shard_index=shard_index,
        skip_examples=skip_examples,
    )
    validation_stream = _build_stream(
        spec,
        {"validation": validation_files},
        num_shards=num_shards,
        shard_index=shard_index,
        skip_examples=skip_examples,
    )
    return train_stream, validation_stream


def get_dataset(*, num_shards=1, shard_index=0, skip_examples=0):
    print("Loading and shuffling datasets independently...")
    stream_pairs = [
        _load_stream(
            spec,
            num_shards=num_shards,
            shard_index=shard_index,
            skip_examples=skip_examples,
        )
        for spec in DATASETS
    ]
    print("Done")
    train_streams = [train_stream for train_stream, _ in stream_pairs]
    validation_streams = [validation_stream for _, validation_stream in stream_pairs]
    return train_streams, validation_streams
class DataLoader:
    def __init__(self, batch_size, seq_len, dataset_streams, buffer_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")
        if not dataset_streams:
            raise ValueError("dataset_streams must not be empty")

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset_streams = list(dataset_streams)
        self.buffer_size = buffer_size
        self._batch_examples = sum(
            stream.examples_per_batch for stream in self.dataset_streams
        )
        if self._batch_examples != self.batch_size:
            raise ValueError(
                f"sum(DATASETS examples_per_batch) must equal batch_size ({self.batch_size}), "
                f"got {self._batch_examples}"
            )

        self._enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self._enc.eot_token

        self._dataset_iters = [iter(stream.dataset) for stream in self.dataset_streams]
        self._token_buffers = [[] for _ in self.dataset_streams]
        self._sample_queues = [queue.Queue(maxsize=8) for _ in self.dataset_streams]
        self._batch_queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self._fetcher_threads = [
            threading.Thread(
                target=self._fetcher_loop,
                args=(dataset_idx,),
                daemon=True,
                name=f"dataloader-fetcher-{dataset_idx}",
            )
            for dataset_idx in range(len(self.dataset_streams))
        ]
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="dataloader-producer"
        )
        for fetcher_thread in self._fetcher_threads:
            fetcher_thread.start()
        self._producer_thread.start()

    def _next_sample(self, dataset_idx):
        while True:
            try:
                return next(self._dataset_iters[dataset_idx])
            except StopIteration:
                # Restart for finite datasets so iteration can continue.
                self._dataset_iters[dataset_idx] = iter(
                    self.dataset_streams[dataset_idx].dataset
                )

    # Sentinel for sample queue when fetcher stops
    _SAMPLE_END = object()

    def _put_with_stop(self, q, item):
        while not self._stop_event.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _fetcher_loop(self, dataset_idx):
        try:
            while not self._stop_event.is_set():
                sample = self._next_sample(dataset_idx)
                self._put_with_stop(self._sample_queues[dataset_idx], sample)
        finally:
            self._put_with_stop(self._sample_queues[dataset_idx], self._SAMPLE_END)

    def _fill_token_buffer(self, dataset_idx, min_tokens):
        token_buffer = self._token_buffers[dataset_idx]
        sample_queue = self._sample_queues[dataset_idx]
        while len(token_buffer) < min_tokens and not self._stop_event.is_set():
            try:
                sample = sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if sample is self._SAMPLE_END:
                return
            text = _extract_text(sample)
            doc_tokens = self._enc.encode_ordinary(text)
            doc_tokens.append(self.eot_token)
            token_buffer.extend(doc_tokens)

    def _build_batch(self):
        x_parts = []
        y_parts = []

        for dataset_idx, stream in enumerate(self.dataset_streams):
            dataset_tokens_per_batch = stream.examples_per_batch * self.seq_len + 1
            self._fill_token_buffer(dataset_idx, dataset_tokens_per_batch)
            token_buffer = self._token_buffers[dataset_idx]
            if len(token_buffer) < dataset_tokens_per_batch:
                raise StopIteration

            flat = token_buffer[:dataset_tokens_per_batch]
            self._token_buffers[dataset_idx] = token_buffer[dataset_tokens_per_batch:]
            flat = torch.tensor(flat, dtype=torch.long)
            x_part = flat[:-1].view(stream.examples_per_batch, self.seq_len)
            y_part = flat[1:].view(stream.examples_per_batch, self.seq_len)
            x_parts.append(x_part)
            y_parts.append(y_part)

        x = torch.cat(x_parts, dim=0)
        y = torch.cat(y_parts, dim=0)
        return x, y

    def _producer_loop(self):
        try:
            while not self._stop_event.is_set():
                batch = self._build_batch()
                # Block only when queue is full (backpressure); otherwise emit immediately
                if not self._put_with_stop(self._batch_queue, batch):
                    return
        except StopIteration:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._stop_event.is_set() and self._batch_queue.empty():
                raise StopIteration
            try:
                x, y = self._batch_queue.get(timeout=0.1)
                return x, y
            except queue.Empty:
                if not self._producer_thread.is_alive():
                    if self._batch_queue.empty():
                        raise StopIteration

    def close(self):
        self._stop_event.set()
        for fetcher_thread in self._fetcher_threads:
            if fetcher_thread.is_alive():
                fetcher_thread.join(timeout=1.0)
        if self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)

    def reset(self):
        self.close()
        self._stop_event = threading.Event()
        self._dataset_iters = [iter(stream.dataset) for stream in self.dataset_streams]
        self._token_buffers = [[] for _ in self.dataset_streams]
        self._sample_queues = [queue.Queue(maxsize=8) for _ in self.dataset_streams]
        self._batch_queue = queue.Queue(maxsize=self.buffer_size)
        self._fetcher_threads = [
            threading.Thread(
                target=self._fetcher_loop,
                args=(dataset_idx,),
                daemon=True,
                name=f"dataloader-fetcher-{dataset_idx}",
            )
            for dataset_idx in range(len(self.dataset_streams))
        ]
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="dataloader-producer"
        )
        for fetcher_thread in self._fetcher_threads:
            fetcher_thread.start()
        self._producer_thread.start()


if __name__ == "__main__":
    dataset_streams, _ = get_dataset()
    loader = DataLoader(
        batch_size=sum(spec.examples_per_batch for spec in DATASETS),
        seq_len=128,
        dataset_streams=dataset_streams,
        buffer_size=2,
    )
    print("Initialized loader")
    try:
        for i in range(5):
            x, y = next(loader)
            print(f"batch {i + 1}: x {x.shape}, y {y.shape}")
    finally:
        loader.close()
