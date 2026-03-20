import queue
import threading
from collections import Counter

import tiktoken
import torch
from datasets import load_dataset, interleave_datasets

DATASETS = [
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", 0.4),
    ("mlfoundations/dclm-baseline-1.0", None, "train", 0.4),
    ("HuggingFaceTB/finemath", "finemath-3plus", "train", 0.1),
    ("HuggingFaceTB/finemath", "infiwebmath-3plus", "train", 0.1),
]


def get_dataset():
    hf_datasets = [
        load_dataset(namespace, split=split, streaming=True)
        if dataset_name is None
        else load_dataset(namespace, name=dataset_name, split=split, streaming=True)
        for namespace, dataset_name, split, _ in DATASETS
    ]
    for i, ds in enumerate(hf_datasets):
        cols_to_remove = [c for c in ds.column_names if c != "text"]
        if cols_to_remove:
            hf_datasets[i] = ds.remove_columns(cols_to_remove).map(
                lambda x: {**x, "source": DATASETS[i][0]}
            )
    print("Interleaving datasets...")
    out = interleave_datasets(hf_datasets, probabilities=[p for _, _, _, p in DATASETS])
    print("Shuffling dataset...")
    out = out.shuffle(seed=42)
    print("Done")
    return out


def _format_source_proportions(source_proportions):
    if not source_proportions:
        return "unknown=1.000"
    return " | ".join(
        f"{source}={proportion:.3f}"
        for source, proportion in sorted(source_proportions.items())
    )


class DataLoader:
    def __init__(self, batch_size, seq_len, dataset, buffer_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be > 0")

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dataset = dataset
        self.buffer_size = buffer_size
        self._tokens_per_batch = self.batch_size * self.seq_len + 1

        self._enc = tiktoken.get_encoding("gpt2")
        self.eot_token = self._enc.eot_token

        self._dataset_iter = iter(self.dataset)
        self._token_buffer = []
        self._source_buffer = []
        self._sample_queue = queue.Queue(maxsize=8)
        self._batch_queue = queue.Queue(maxsize=self.buffer_size)
        self._stop_event = threading.Event()
        self._fetcher_thread = threading.Thread(
            target=self._fetcher_loop, daemon=True, name="dataloader-fetcher"
        )
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="dataloader-producer"
        )
        self._fetcher_thread.start()
        self._producer_thread.start()

    def _extract_text_and_source(self, sample):
        if isinstance(sample, str):
            return sample, "unknown"
        if isinstance(sample, dict):
            if "text" in sample and isinstance(sample["text"], str):
                return sample["text"], sample.get("source", "unknown")
            raise ValueError("dataset dict samples must include a string 'text' field")
        raise TypeError("dataset samples must be strings or dicts with a 'text' field")

    def _next_sample(self):
        while True:
            try:
                return next(self._dataset_iter)
            except StopIteration:
                # Restart for finite datasets so iteration can continue.
                self._dataset_iter = iter(self.dataset)

    # Sentinel for sample queue when fetcher stops
    _SAMPLE_END = object()

    def _fetcher_loop(self):
        try:
            while not self._stop_event.is_set():
                sample = self._next_sample()
                self._sample_queue.put(sample)
        finally:
            self._sample_queue.put(self._SAMPLE_END)

    def _fill_token_buffer(self, min_tokens):
        while len(self._token_buffer) < min_tokens and not self._stop_event.is_set():
            try:
                sample = self._sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if sample is self._SAMPLE_END:
                return
            text, source = self._extract_text_and_source(sample)
            doc_tokens = self._enc.encode_ordinary(text)
            doc_tokens.append(self.eot_token)
            self._token_buffer.extend(doc_tokens)
            self._source_buffer.extend([source] * len(doc_tokens))

    def _build_batch(self):
        self._fill_token_buffer(self._tokens_per_batch)
        if len(self._token_buffer) < self._tokens_per_batch:
            raise StopIteration

        flat = self._token_buffer[: self._tokens_per_batch]
        flat_sources = self._source_buffer[: self._tokens_per_batch]
        self._token_buffer = self._token_buffer[self._tokens_per_batch :]
        self._source_buffer = self._source_buffer[self._tokens_per_batch :]

        flat = torch.tensor(flat, dtype=torch.long)
        x = flat[:-1].view(self.batch_size, self.seq_len)
        y = flat[1:].view(self.batch_size, self.seq_len)
        source_counts = Counter(flat_sources[:-1])
        total_tokens = max(sum(source_counts.values()), 1)
        source_proportions = {
            source: count / total_tokens
            for source, count in sorted(source_counts.items())
        }
        return x, y, source_proportions

    def _producer_loop(self):
        try:
            while not self._stop_event.is_set():
                batch = self._build_batch()
                # Block only when queue is full (backpressure); otherwise emit immediately
                self._batch_queue.put(batch)
        except StopIteration:
            pass

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if self._stop_event.is_set() and self._batch_queue.empty():
                raise StopIteration
            try:
                x, y, source_proportions = self._batch_queue.get(timeout=0.1)
                print(
                    f"batch sources {_format_source_proportions(source_proportions)}"
                )
                return x, y
            except queue.Empty:
                if not self._producer_thread.is_alive():
                    if self._batch_queue.empty():
                        raise StopIteration

    def close(self):
        self._stop_event.set()
        if self._fetcher_thread.is_alive():
            self._fetcher_thread.join(timeout=1.0)
        if self._producer_thread.is_alive():
            self._producer_thread.join(timeout=1.0)

    def reset(self):
        self.close()
        self._stop_event = threading.Event()
        self._dataset_iter = iter(self.dataset)
        self._token_buffer = []
        self._source_buffer = []
        self._sample_queue = queue.Queue(maxsize=8)
        self._batch_queue = queue.Queue(maxsize=self.buffer_size)
        self._fetcher_thread = threading.Thread(
            target=self._fetcher_loop, daemon=True, name="dataloader-fetcher"
        )
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="dataloader-producer"
        )
        self._fetcher_thread.start()
        self._producer_thread.start()


if __name__ == "__main__":
    dataset = get_dataset()
    loader = DataLoader(batch_size=4, seq_len=128, dataset=dataset, buffer_size=2)
    print("Initialized loader")
    try:
        for i in range(5):
            x, y = next(loader)
            print(f"batch {i + 1}: x {x.shape}, y {y.shape}")
    finally:
        loader.close()
