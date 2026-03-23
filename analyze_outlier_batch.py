import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
import torch
from torch.nn import functional as F

from model import Model, ModelConfig, PositionEmbeddingType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a saved outlier batch through the model and sort tokens by loss."
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
    model.to("cuda")
    return model


def escape_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def token_text(tokenizer, token_id: int) -> str:
    return escape_text(tokenizer.decode([token_id]))


def total_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    grad_norms = [
        torch.linalg.vector_norm(parameter.grad.detach().float(), 2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not grad_norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(grad_norms), 2))


def analyze_checkpoint(
    checkpoint_path: Path, top_k: int, context_window: int, output_path: str | None
) -> None:
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(checkpoint)
    tokenizer = tiktoken.get_encoding("gpt2")
    eot_token_id = tokenizer.eot_token
    saved_args = checkpoint["args"]
    batch_contents = checkpoint.get("outlier_batch_contents")
    autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    grad_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    rows: list[dict[str, Any]] = []
    for micro_batch_idx, (x, y) in enumerate(batch_contents):
        x = x.to(dtype=torch.long, device="cuda")
        targets = y.to(dtype=torch.long, device="cuda")
        for sample_idx in range(x.size(0)):
            sample_x = x[sample_idx : sample_idx + 1]
            sample_targets = targets[sample_idx : sample_idx + 1]
            valid_mask = sample_targets.ne(-1)
            attn_mask = None
            if saved_args.get("intra_doc_mask", False):
                attn_mask = make_intra_document_attn_mask(sample_x, eot_token_id)

            with autocast_context:
                logits, _ = model(sample_x, attn_mask=attn_mask)
                token_losses = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    sample_targets.view(-1),
                    reduction="none",
                    ignore_index=-1,
                ).view_as(sample_x)
            valid_token_losses = token_losses[valid_mask]
            if valid_token_losses.numel() == 0:
                continue

            mean_loss = valid_token_losses.mean()
            model.zero_grad(set_to_none=True)
            mean_loss.backward()
            grad_norm = total_grad_norm(grad_parameters)
            model.zero_grad(set_to_none=True)

            token_positions: list[int] = []
            token_loss_values: list[float] = []
            input_token_ids: list[int] = []
            target_token_ids: list[int] = []
            input_tokens: list[str] = []
            target_tokens: list[str] = []
            contexts: list[str] = []
            for position_idx in range(sample_x.size(1)):
                if not valid_mask[0, position_idx]:
                    continue

                start = max(0, position_idx - context_window)
                context_ids = sample_x[0, start : position_idx + 1].tolist()
                input_id = int(sample_x[0, position_idx])
                target_id = int(sample_targets[0, position_idx])
                token_positions.append(position_idx)
                token_loss_values.append(float(token_losses[0, position_idx]))
                input_token_ids.append(input_id)
                target_token_ids.append(target_id)
                input_tokens.append(token_text(tokenizer, input_id))
                target_tokens.append(token_text(tokenizer, target_id))
                contexts.append(escape_text(tokenizer.decode(context_ids)))

            rows.append(
                {
                    "loss": float(mean_loss.detach()),
                    "grad_norm": grad_norm,
                    "micro_batch": micro_batch_idx,
                    "sample": sample_idx,
                    "token_count": len(token_loss_values),
                    "token_positions": token_positions,
                    "token_losses": token_loss_values,
                    "input_token_ids": input_token_ids,
                    "target_token_ids": target_token_ids,
                    "input_tokens": input_tokens,
                    "target_tokens": target_tokens,
                    "contexts": contexts,
                }
            )

    rows.sort(key=lambda row: row["loss"], reverse=True)

    print(f"checkpoint: {checkpoint_path}")
    print(f"step: {checkpoint.get('step')}")
    print(f"saved loss: {checkpoint.get('loss')}")
    print(f"saved grad norm: {checkpoint.get('grad_norm')}")
    print(f"micro-batches: {len(batch_contents)}")
    print(f"scored examples: {len(rows)}")
    if checkpoint.get("outlier_batch_contents") is None:
        print(
            "note: this checkpoint uses the older x-only format, so the final "
            "flattened token of each saved micro-batch is skipped."
        )
    print()
    print("top example losses:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:4d} | loss={row['loss']:.6f} | grad_norm={row['grad_norm']:.6f} | "
            f"micro={row['micro_batch']} | sample={row['sample']} | "
            f"tokens={row['token_count']}"
        )

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(rows, start=1):
            output_rows.append(
                {
                    "rank": rank,
                    "loss": row["loss"],
                    "grad_norm": row["grad_norm"],
                    "micro_batch": row["micro_batch"],
                    "sample": row["sample"],
                    "token_count": row["token_count"],
                    "token_positions": json.dumps(
                        row["token_positions"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "token_losses": json.dumps(
                        row["token_losses"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "input_token_ids": json.dumps(
                        row["input_token_ids"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "target_token_ids": json.dumps(
                        row["target_token_ids"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "input_tokens": json.dumps(
                        row["input_tokens"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "target_tokens": json.dumps(
                        row["target_tokens"], ensure_ascii=True, separators=(",", ":")
                    ),
                    "contexts": json.dumps(
                        row["contexts"], ensure_ascii=True, separators=(",", ":")
                    ),
                }
            )
        pd.DataFrame(output_rows).to_csv(output, sep="\t", index=False)
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
