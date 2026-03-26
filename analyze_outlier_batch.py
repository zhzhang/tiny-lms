import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import tiktoken
import torch
from torch.nn import functional as F

from model import Model, ModelConfig, PositionEmbeddingType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run saved outlier batches through the model and sort examples by loss."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help=(
            "Path to an outlier checkpoint or directory. Defaults to scanning every "
            "*_outlier.pt under outlier_batches/."
        ),
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
        help="Optional path to write the full sorted results as a pandas pickle.",
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


def resolve_checkpoint_paths(path_arg: str | None) -> list[Path]:
    if path_arg is not None:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint or directory not found: {path}")
        if path.is_file():
            return [path]
        matches = sorted(path.rglob("*_outlier.pt"), key=lambda checkpoint: checkpoint.stat().st_mtime)
    else:
        matches = sorted(
            Path("outlier_batches").rglob("*_outlier.pt"),
            key=lambda checkpoint: checkpoint.stat().st_mtime,
        )
    if not matches:
        search_root = path_arg if path_arg is not None else "outlier_batches/"
        raise FileNotFoundError(f"No outlier checkpoint found under {search_root}")
    return matches


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
    checkpoint_path: Path, context_window: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_name": checkpoint_path.name,
                    "checkpoint_step": checkpoint.get("step"),
                    "checkpoint_saved_loss": checkpoint.get("loss"),
                    "checkpoint_saved_grad_norm": checkpoint.get("grad_norm"),
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
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "step": checkpoint.get("step"),
        "saved_loss": checkpoint.get("loss"),
        "saved_grad_norm": checkpoint.get("grad_norm"),
        "micro_batch_count": len(batch_contents),
        "scored_examples": len(rows),
    }
    return rows, summary


def print_summary(
    checkpoint_summaries: list[dict[str, Any]], rows: list[dict[str, Any]], top_k: int
) -> None:
    print(f"checkpoints analyzed: {len(checkpoint_summaries)}")
    print(f"scored examples: {len(rows)}")
    print()
    print("per-checkpoint summary:")
    for summary in checkpoint_summaries:
        print(
            f"{summary['checkpoint_path']} | step={summary['step']} | "
            f"saved_loss={summary['saved_loss']} | "
            f"saved_grad_norm={summary['saved_grad_norm']} | "
            f"micro_batches={summary['micro_batch_count']} | "
            f"scored_examples={summary['scored_examples']}"
        )
    print()
    print("top example losses:")
    for rank, row in enumerate(rows[:top_k], start=1):
        print(
            f"{rank:4d} | loss={row['loss']:.6f} | grad_norm={row['grad_norm']:.6f} | "
            f"checkpoint={row['checkpoint_name']} | micro={row['micro_batch']} | "
            f"sample={row['sample']} | tokens={row['token_count']}"
        )


def build_output_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    dataframe.insert(0, "rank", range(1, len(dataframe) + 1))
    return dataframe


def analyze_checkpoints(
    checkpoint_paths: list[Path], top_k: int, context_window: int, output_path: str | None
) -> None:
    all_rows: list[dict[str, Any]] = []
    checkpoint_summaries: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        checkpoint_rows, checkpoint_summary = analyze_checkpoint(
            checkpoint_path=checkpoint_path,
            context_window=context_window,
        )
        all_rows.extend(checkpoint_rows)
        checkpoint_summaries.append(checkpoint_summary)

    all_rows.sort(key=lambda row: row["loss"], reverse=True)
    print_summary(checkpoint_summaries, all_rows, top_k)

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        build_output_dataframe(all_rows).to_pickle(output)
        print()
        print(f"wrote sorted pickle: {output}")


if __name__ == "__main__":
    args = parse_args()
    analyze_checkpoints(
        checkpoint_paths=resolve_checkpoint_paths(args.checkpoint),
        top_k=args.top_k,
        context_window=args.context_window,
        output_path=args.output,
    )
