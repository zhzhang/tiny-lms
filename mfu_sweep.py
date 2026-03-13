import argparse
import csv
import gc
import time
from pathlib import Path

import torch

from model import AttentionHead, ModelConfig, PositionEmbeddingType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep attention-head throughput across d_model, context length, and "
            "batch size. Records per-batch processing time normalized by tokens."
        )
    )
    parser.add_argument("--d-model-start", type=int, default=512)
    parser.add_argument("--d-model-end", type=int, default=4096)
    parser.add_argument("--ctx-start", type=int, default=1024)
    parser.add_argument("--ctx-end", type=int, default=4096)
    parser.add_argument("--batches-per-size", type=int, default=10)
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("mfu_sweep_results.csv"),
        help="CSV output path.",
    )
    return parser.parse_args()


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or isinstance(exc, torch.cuda.OutOfMemoryError)


def to_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def build_compiled_attention_head(
    d_model: int,
    context_length: int,
    device: torch.device,
    dtype: torch.dtype,
):
    config = ModelConfig(
        d_model=d_model,
        n_kv_heads=1,
        n_q_heads=1,
        n_layers=1,
        max_sequence_length=context_length,
        vocab_size=1,
        position_embedding_type=PositionEmbeddingType.NOPE,
    )
    module = AttentionHead(config).to(device=device, dtype=dtype).eval()
    # Use dynamic=True because batch size changes during the sweep.
    compiled = torch.compile(module, dynamic=True)
    return compiled


@torch.inference_mode()
def run_single_batch(
    compiled_head: torch.nn.Module,
    batch_size: int,
    context_length: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    x = torch.randn(
        batch_size,
        context_length,
        d_model,
        device=device,
        dtype=dtype,
    )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _ = compiled_head(x)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_s = start.elapsed_time(end) / 1000.0
    del x
    return elapsed_s


def clear_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def ensure_csv_header(path: Path) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        return
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "d_model",
                "context_length",
                "batch_size",
                "batch_idx",
                "elapsed_seconds",
                "tokens",
                "seconds_per_token",
            ]
        )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep, but no GPU was found.")

    dtype = to_dtype(args.dtype)
    device = torch.device("cuda")
    ensure_csv_header(args.output_csv)

    with args.output_csv.open("a", newline="") as f:
        writer = csv.writer(f)

        for d_model in range(args.d_model_start, args.d_model_end + 1):
            for context_length in range(args.ctx_start, args.ctx_end + 1):
                print(
                    f"Building+compiling attention head: d_model={d_model}, ctx={context_length}"
                )

                try:
                    clear_cuda_memory()
                    compiled_head = build_compiled_attention_head(
                        d_model=d_model,
                        context_length=context_length,
                        device=device,
                        dtype=dtype,
                    )
                    # Warmup compile/runtime with batch size 1.
                    _ = run_single_batch(
                        compiled_head=compiled_head,
                        batch_size=1,
                        context_length=context_length,
                        d_model=d_model,
                        device=device,
                        dtype=dtype,
                    )
                except Exception as exc:
                    if is_oom_error(exc):
                        print(
                            f"OOM while building/compiling model for d_model={d_model}, "
                            f"ctx={context_length}; skipping."
                        )
                        clear_cuda_memory()
                        continue
                    raise

                batch_size = 1
                while True:
                    print(
                        f"Timing: d_model={d_model}, ctx={context_length}, batch={batch_size}, "
                        f"runs={args.batches_per_size}"
                    )
                    try:
                        for batch_idx in range(args.batches_per_size):
                            elapsed_s = run_single_batch(
                                compiled_head=compiled_head,
                                batch_size=batch_size,
                                context_length=context_length,
                                d_model=d_model,
                                device=device,
                                dtype=dtype,
                            )
                            tokens = batch_size * context_length
                            seconds_per_token = elapsed_s / tokens
                            writer.writerow(
                                [
                                    d_model,
                                    context_length,
                                    batch_size,
                                    batch_idx,
                                    f"{elapsed_s:.9f}",
                                    tokens,
                                    f"{seconds_per_token:.12e}",
                                ]
                            )
                        f.flush()
                        batch_size += 1
                        if batch_size > 64:
                            break
                    except Exception as exc:
                        if is_oom_error(exc):
                            print(
                                "OOM at "
                                f"d_model={d_model}, ctx={context_length}, batch={batch_size}. "
                                "Moving to next (d_model, context_length)."
                            )
                            clear_cuda_memory()
                            break
                        raise

                del compiled_head
                clear_cuda_memory()
                # Small pause to make GPU memory behavior less bursty between configs.
                time.sleep(0.01)


if __name__ == "__main__":
    main()
