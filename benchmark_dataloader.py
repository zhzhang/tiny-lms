import argparse
import time

from dataloader import DATASETS, DataLoader, get_dataset


def parse_args():
    default_batch_size = sum(spec.examples_per_batch for spec in DATASETS)
    parser = argparse.ArgumentParser(
        description="Benchmark train dataloader throughput."
    )
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--data-buffer-size", type=int, default=128)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-examples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset_streams, _ = get_dataset(
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        skip_examples=args.skip_examples,
    )
    loader = DataLoader(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        dataset_streams=train_dataset_streams,
        buffer_size=args.data_buffer_size,
    )

    total_tokens = 0
    per_batch_toks_per_s = []
    benchmark_start = time.perf_counter()

    try:
        for batch_idx in range(args.num_batches):
            batch_start = time.perf_counter()
            x, _ = next(loader)
            batch_elapsed = time.perf_counter() - batch_start
            batch_tokens = x.numel()
            total_tokens += batch_tokens

            toks_per_s = batch_tokens / batch_elapsed
            per_batch_toks_per_s.append(toks_per_s)
            print(
                f"batch {batch_idx + 1:3d}/{args.num_batches} | "
                f"{batch_tokens} toks | {batch_elapsed * 1000:.2f} ms | "
                f"{toks_per_s:.2f} toks/s"
            )
    finally:
        loader.close()

    total_elapsed = time.perf_counter() - benchmark_start
    mean_batch_toks_per_s = sum(per_batch_toks_per_s) / len(per_batch_toks_per_s)

    print(
        "\nsummary | "
        f"batches {args.num_batches} | "
        f"tokens {total_tokens} | "
        f"elapsed {total_elapsed:.2f} s | "
        f"mean_batch {mean_batch_toks_per_s:.2f} toks/s"
    )


if __name__ == "__main__":
    main()
