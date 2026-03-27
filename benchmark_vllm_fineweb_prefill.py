#!/usr/bin/env python3
"""Benchmark vLLM prefill latency on FineWeb prefixes with Qwen 7B.

This script is intentionally self-contained and does not depend on the rest of
the repository.

Example:
    uv run benchmark_vllm_fineweb_prefill.py

If `vllm` is not installed yet:
    uv add vllm
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected in this repo
    torch = None  # type: ignore[assignment]

try:
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - import error is user-facing
    raise SystemExit(
        "vllm is required for this benchmark. Install it with `uv add vllm` "
        "and then rerun the script."
    ) from exc


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATASET = "HuggingFaceFW/fineweb"
DEFAULT_FINEWEB_CONFIG = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT = "vllm_fineweb_qwen7b_prefill.tsv"


@dataclass
class BenchmarkRow:
    index: int
    prompt_tokens: int
    prefill_seconds: float
    wall_clock_seconds: float
    timing_source: str
    queued_seconds: float | None
    inference_seconds: float | None
    decode_seconds: float | None
    e2e_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream FineWeb, build a 100k-token sample, and benchmark vLLM "
            "prefill latency over 100 log-spaced prompt lengths."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--fineweb-config", default=DEFAULT_FINEWEB_CONFIG)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--min-prompt-tokens", type=int, default=10)
    parser.add_argument("--max-prompt-tokens", type=int, default=100_000)
    parser.add_argument("--num-points", type=int, default=100)
    parser.add_argument("--separator", default="\n\n")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=100_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def log_spaced_ints(start: int, stop: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("num_points must be at least 1")
    if count < 2:
        return [start]
    if start < 1 or stop < start:
        raise ValueError("prompt token range must satisfy 1 <= start <= stop")
    if count > (stop - start + 1):
        raise ValueError("cannot generate strictly increasing integer sizes")

    sizes: list[int] = []
    last = start - 1
    log_start = math.log(start)
    log_stop = math.log(stop)

    for idx in range(count):
        frac = idx / (count - 1)
        raw_value = math.exp(log_start + frac * (log_stop - log_start))
        candidate = int(round(raw_value))
        min_allowed = last + 1
        max_allowed = stop - (count - idx - 1)
        candidate = max(min_allowed, min(candidate, max_allowed))
        sizes.append(candidate)
        last = candidate

    sizes[0] = start
    sizes[-1] = stop
    return sizes


def synchronize_cuda() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_fineweb_prefix_tokens(
    *,
    dataset_name: str,
    config_name: str,
    split: str,
    tokenizer: Any,
    target_tokens: int,
    separator: str,
) -> tuple[list[int], int]:
    dataset = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=True,
    )
    separator_ids = tokenizer.encode(separator, add_special_tokens=False)
    collected: list[int] = []
    source_examples = 0

    for record in dataset:
        text = record.get("text")
        if not text:
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            continue

        if collected and separator_ids:
            collected.extend(separator_ids)
        collected.extend(token_ids)
        source_examples += 1

        if len(collected) >= target_tokens:
            return collected[:target_tokens], source_examples

    raise RuntimeError(
        f"FineWeb stream ended before reaching {target_tokens} prompt tokens."
    )


def maybe_get_metric(obj: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def extract_prefill_seconds(output: Any, wall_clock_seconds: float) -> tuple[float, str]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return wall_clock_seconds, "wall_clock_generate"

    direct_prefill = maybe_get_metric(
        metrics,
        "prefill_time",
        "prefill_seconds",
        "request_prefill_time_seconds",
    )
    if direct_prefill is not None:
        return direct_prefill, "metrics.prefill_time"

    scheduled_ts = maybe_get_metric(metrics, "scheduled_ts")
    first_token_ts = maybe_get_metric(metrics, "first_token_ts")
    if scheduled_ts is not None and first_token_ts is not None:
        delta = first_token_ts - scheduled_ts
        if delta >= 0:
            return delta, "metrics.first_token_ts - metrics.scheduled_ts"

    return wall_clock_seconds, "wall_clock_generate"


def reset_prefix_cache(llm: LLM) -> str:
    candidates = [
        ("llm.reset_prefix_cache", getattr(llm, "reset_prefix_cache", None)),
        (
            "llm.llm_engine.reset_prefix_cache",
            getattr(getattr(llm, "llm_engine", None), "reset_prefix_cache", None),
        ),
        (
            "llm.engine.reset_prefix_cache",
            getattr(getattr(llm, "engine", None), "reset_prefix_cache", None),
        ),
    ]
    for label, method in candidates:
        if callable(method):
            reset_result = method()
            if reset_result is False:
                raise RuntimeError(
                    f"{label} reported failure while clearing the prefix cache."
                )
            synchronize_cuda()
            return label

    raise RuntimeError(
        "The installed vLLM build does not expose `reset_prefix_cache()`. "
        "Please upgrade vLLM to a version that supports prefix cache reset so "
        "the benchmark is not polluted by cache reuse."
    )


def create_llm(args: argparse.Namespace) -> LLM:
    return LLM(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max(args.max_model_len, args.max_prompt_tokens),
        max_num_seqs=1,
        enable_prefix_caching=True,
        disable_log_stats=False,
        trust_remote_code=args.trust_remote_code,
    )


def benchmark_prefill(
    *,
    llm: LLM,
    prompt_prefix_tokens: list[int],
    prompt_sizes: list[int],
) -> list[BenchmarkRow]:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        ignore_eos=True,
    )

    warmup_prompt = {"prompt_token_ids": prompt_prefix_tokens[: max(10, prompt_sizes[0])]}
    llm.generate([warmup_prompt], sampling_params, use_tqdm=False)
    cache_reset_impl = reset_prefix_cache(llm)
    print(f"Using prefix cache reset via: {cache_reset_impl}")

    rows: list[BenchmarkRow] = []
    for index, prompt_tokens in enumerate(prompt_sizes, start=1):
        prompt = {"prompt_token_ids": prompt_prefix_tokens[:prompt_tokens]}

        reset_prefix_cache(llm)
        synchronize_cuda()
        started_at = time.perf_counter()
        output = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
        synchronize_cuda()
        wall_clock_seconds = time.perf_counter() - started_at

        prefill_seconds, timing_source = extract_prefill_seconds(
            output,
            wall_clock_seconds,
        )
        metrics = getattr(output, "metrics", None)

        row = BenchmarkRow(
            index=index,
            prompt_tokens=prompt_tokens,
            prefill_seconds=prefill_seconds,
            wall_clock_seconds=wall_clock_seconds,
            timing_source=timing_source,
            queued_seconds=maybe_get_metric(metrics, "queued_time", "queued_seconds"),
            inference_seconds=maybe_get_metric(
                metrics,
                "inference_time",
                "inference_seconds",
            ),
            decode_seconds=maybe_get_metric(metrics, "decode_time", "decode_seconds"),
            e2e_seconds=maybe_get_metric(metrics, "e2e_latency", "e2e_seconds"),
        )
        rows.append(row)

        print(
            f"[{index:03d}/{len(prompt_sizes):03d}] "
            f"prompt_tokens={prompt_tokens:>6d} "
            f"prefill_seconds={prefill_seconds:.6f} "
            f"source={timing_source}"
        )

    return rows


def write_results(rows: list[BenchmarkRow], output_path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    if args.max_prompt_tokens > args.max_model_len:
        args.max_model_len = args.max_prompt_tokens

    prompt_sizes = log_spaced_ints(
        args.min_prompt_tokens,
        args.max_prompt_tokens,
        args.num_points,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )

    print(
        "Streaming FineWeb and building one shared prefix corpus up to "
        f"{args.max_prompt_tokens} tokens..."
    )
    prefix_tokens, source_examples = build_fineweb_prefix_tokens(
        dataset_name=args.dataset,
        config_name=args.fineweb_config,
        split=args.split,
        tokenizer=tokenizer,
        target_tokens=args.max_prompt_tokens,
        separator=args.separator,
    )
    print(
        f"Collected {len(prefix_tokens)} tokens from {source_examples} FineWeb "
        "examples."
    )

    print(f"Loading vLLM model: {args.model}")
    llm = create_llm(args)
    rows = benchmark_prefill(
        llm=llm,
        prompt_prefix_tokens=prefix_tokens,
        prompt_sizes=prompt_sizes,
    )

    output_path = Path(args.output)
    write_results(rows, output_path)
    print(f"Wrote {len(rows)} benchmark rows to {output_path}")


if __name__ == "__main__":
    main()
