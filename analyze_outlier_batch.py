import argparse
from pathlib import Path
from typing import Any

import tiktoken
import torch
from torch.nn import functional as F

from model import Model, ModelConfig, PositionEmbeddingType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved outlier batch through the model on CPU and sort tokens by loss."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="Path to an outlier checkpoint. Defaults to the newest file in outlier_batches/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many highest-loss token rows to print.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1024,
        help="How many prior input tokens to include in the printed context.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the full sorted results as TSV.",
    )
    return parser.parse_args()


def make_intra_document_attn_mask(tokens: torch.Tensor, eot_token: int) -> torch.Tensor:
    _, seq_len = tokens.shape
    eot_hits = tokens.eq(eot_token)
    doc_ids = torch.nn.functional.pad(eot_hits.cumsum(dim=1)[:, :-1], (1, 0), value=0)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])
    causal = torch.ones(
        (seq_len, seq_len), device=tokens.device, dtype=torch.bool
    ).tril()
    return (same_doc & causal).unsqueeze(1)


def resolve_checkpoint_path(path_arg: str | None) -> Path:
    if path_arg is not None:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    matches = sorted(
        Path("outlier_batches").rglob("*_outlier.pt"),
        key=lambda path: path.stat().st_mtime,
    )
    if not matches:
        raise FileNotFoundError("No outlier checkpoint found under outlier_batches/")
    return matches[-1]


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def coerce_position_embedding_type(value: Any) -> PositionEmbeddingType:
    if isinstance(value, PositionEmbeddingType):
        return value
    if isinstance(value, str):
        return PositionEmbeddingType(value)
    raise TypeError(f"Unsupported position embedding type: {value!r}")


def build_model(checkpoint: dict[str, Any]) -> Model:
    saved_args = checkpoint["args"]
    config = ModelConfig(
        d_model=saved_args["d_model"],
        n_kv_heads=saved_args["n_kv_heads"],
        n_q_heads=saved_args["n_q_heads"],
        n_layers=saved_args["n_layers"],
        max_sequence_length=saved_args["max_sequence_length"],
        vocab_size=saved_args["vocab_size"],
        position_embedding_type=coerce_position_embedding_type(
            saved_args["position_embedding_type"]
        ),
        rope_skip_freq=saved_args.get("rope_skip_freq", 2),
        no_bias=saved_args.get("no_bias", False),
    )
    model = Model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to("cpu")
    return model


def escape_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def token_text(tokenizer, token_id: int) -> str:
    return escape_text(tokenizer.decode([token_id]))


def build_targets_from_x(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat_x = x.reshape(-1)
    flat_y = flat_x.new_full(flat_x.shape, -1)
    flat_y[:-1] = flat_x[1:]
    valid = flat_y.ne(-1)
    return flat_y.view_as(x), valid.view_as(x)


def analyze_checkpoint(
    checkpoint_path: Path, top_k: int, context_window: int, output_path: str | None
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(checkpoint)
    tokenizer = tiktoken.get_encoding("gpt2")
    eot_token_id = tokenizer.eot_token
    saved_args = checkpoint["args"]
    batch_xs = checkpoint["outlier_batch_xs"]

    rows: list[dict[str, Any]] = []
    print(batch_xs)
    with torch.no_grad():
        for micro_batch_idx, x_cpu in enumerate(batch_xs):
            x = x_cpu.to(dtype=torch.long, device="cpu")
            targets, valid_mask = build_targets_from_x(x)
            attn_mask = None
            if saved_args.get("intra_doc_mask", False):
                attn_mask = make_intra_document_attn_mask(x, eot_token_id)

            logits, _ = model(x, attn_mask=attn_mask)
            token_losses = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
                ignore_index=-1,
            ).view_as(x)

            for sample_idx in range(x.size(0)):
                for position_idx in range(x.size(1)):
                    if not valid_mask[sample_idx, position_idx]:
                        continue

                    start = max(0, position_idx - context_window)
                    context_ids = x[sample_idx, start : position_idx + 1].tolist()
                    input_id = int(x[sample_idx, position_idx])
                    target_id = int(targets[sample_idx, position_idx])
                    rows.append(
                        {
                            "loss": float(token_losses[sample_idx, position_idx]),
                            "micro_batch": micro_batch_idx,
                            "sample": sample_idx,
                            "position": position_idx,
                            "input_token_id": input_id,
                            "target_token_id": target_id,
                            "input_token": token_text(tokenizer, input_id),
                            "target_token": token_text(tokenizer, target_id),
                            "context": escape_text(tokenizer.decode(context_ids)),
                        }
                    )

    rows.sort(key=lambda row: row["loss"], reverse=True)

    print(f"checkpoint: {checkpoint_path}")
    print(f"step: {checkpoint.get('step')}")
    print(f"saved loss: {checkpoint.get('loss')}")
    print(f"saved grad norm: {checkpoint.get('grad_norm')}")
    print(f"micro-batches: {len(batch_xs)}")
    print(f"scored tokens: {len(rows)}")
    print(
        "note: the final flattened token of each saved micro-batch is skipped because "
        "the outlier checkpoint stores x but not the final next-token target."
    )
    print()
    print("top token losses:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:4d} | loss={row['loss']:.6f} | micro={row['micro_batch']} | "
            f"sample={row['sample']} | pos={row['position']} | "
            f"input={row['input_token_id']}:{row['input_token']} | "
            f"target={row['target_token_id']}:{row['target_token']} | "
            f"context={row['context']}"
        )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            handle.write(
                "\t".join(
                    [
                        "rank",
                        "loss",
                        "micro_batch",
                        "sample",
                        "position",
                        "input_token_id",
                        "target_token_id",
                        "input_token",
                        "target_token",
                        "context",
                    ]
                )
                + "\n"
            )
            for rank, row in enumerate(rows, start=1):
                handle.write(
                    "\t".join(
                        [
                            str(rank),
                            f"{row['loss']:.9f}",
                            str(row["micro_batch"]),
                            str(row["sample"]),
                            str(row["position"]),
                            str(row["input_token_id"]),
                            str(row["target_token_id"]),
                            row["input_token"],
                            row["target_token"],
                            row["context"],
                        ]
                    )
                    + "\n"
                )
        print()
        print(f"wrote sorted TSV: {output}")


if __name__ == "__main__":
    args = parse_args()
    analyze_checkpoint(
        checkpoint_path=resolve_checkpoint_path(args.checkpoint),
        top_k=args.top_k,
        context_window=args.context_window,
        output_path=args.output,
    )
