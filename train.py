import argparse
import os
import re
import time
from collections import deque
from contextlib import nullcontext

import tiktoken
import torch
from dataloader import DataLoader, get_dataset
import lm_eval
from model import Model, ModelConfig, PositionEmbeddingType
from torch import distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT-style model.")
    parser.add_argument(
        "--train-bin-pattern", type=str, default="fineweb10B/fineweb_train_*.bin"
    )
    parser.add_argument(
        "--val-bin-pattern", type=str, default="fineweb10B/fineweb_val_*.bin"
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--data-buffer-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-decay-frac", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--decay-steps", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=20000)
    parser.add_argument("--grad-accum-steps", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-kv-heads", type=int, default=12)
    parser.add_argument("--n-q-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument(
        "--position-embedding-type",
        type=PositionEmbeddingType,
        default=PositionEmbeddingType.LEARNED,
    )
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="gpt2-training")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-log-interval", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--intra-doc-mask", action="store_true")
    return parser.parse_args()


def make_intra_document_attn_mask(tokens: torch.Tensor, eot_token: int) -> torch.Tensor:
    """
    Builds a boolean attention mask of shape (B, T, T) that is causal and
    prevents cross-document attention. A new document starts *after* each EOT.
    """
    _, seq_len = tokens.shape
    eot_hits = tokens.eq(eot_token)
    # Number of EOT delimiters strictly before each position.
    doc_ids = torch.nn.functional.pad(eot_hits.cumsum(dim=1)[:, :-1], (1, 0), value=0)
    same_doc = doc_ids[:, :, None].eq(doc_ids[:, None, :])
    causal = torch.ones(
        (seq_len, seq_len), device=tokens.device, dtype=torch.bool
    ).tril()
    output = same_doc & causal
    return output.unsqueeze(1)


def _save_checkpoint_if_needed(
    *,
    step,
    args,
    rank,
    checkpoint_run_name,
    model,
    is_distributed,
    optimizer,
):
    if args.checkpoint_every <= 0 or (step + 1) % args.checkpoint_every != 0:
        return
    if rank != 0:
        return

    step_num = step + 1
    if checkpoint_run_name is not None:
        checkpoint_path = f"checkpoints/{checkpoint_run_name}/step_{step_num:06d}.pt"
    else:
        checkpoint_path = f"checkpoints/checkpoint_step_{step_num:06d}.pt"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    checkpoint_model = model.module if is_distributed else model
    checkpoint_model = getattr(checkpoint_model, "_orig_mod", checkpoint_model)
    checkpoint = {
        "step": step_num,
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)
    print(
        f"step {step_num:4d}/{args.num_iterations} | saved checkpoint {checkpoint_path}"
    )


def get_wsd_lr(
    it,
    *,
    learning_rate,
    min_lr,
    warmup_iters,
    num_iterations,
    decay_steps,
):
    # 1) linear warmup for warmup_iters steps
    if warmup_iters > 0 and it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    # 2) if it is beyond the schedule horizon, keep the minimum learning rate
    if it > num_iterations:
        return min_lr

    # 3) optional stable plateau if decay starts near the end
    if decay_steps <= 0:
        return learning_rate
    decay_start = max(num_iterations - decay_steps, warmup_iters)
    if it < decay_start:
        return learning_rate

    # 4) linearly decay to min_lr in the final decay region
    decay_denom = max(num_iterations - decay_start, 1)
    decay_ratio = (it - decay_start) / decay_denom
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    return learning_rate - decay_ratio * (learning_rate - min_lr)


ROLLING_METRIC_WINDOW = 100
ROLLING_OUTLIER_FRAC = 0.2
OUTLIER_LOG_PATH = "outlier_batches.log"


def _is_outside_rolling_window(value: float, history: deque, *, frac: float) -> bool:
    """True if value lies more than ``frac`` times the span beyond min/max of history."""
    if len(history) < history.maxlen:
        return False
    lo, hi = min(history), max(history)
    span = max(hi - lo, 1e-12)
    margin = frac * span
    return value < lo - margin or value > hi + margin


def _log_outlier_batches(
    path: str,
    *,
    step_display: int,
    lossf: float,
    grad_norm: float,
    tokenizer,
    batch_xs: list,
) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n")
        f.write(
            f"step {step_display} | loss {lossf:.6f} | grad_norm {grad_norm:.6f}\n"
        )
        for mi, x_cpu in enumerate(batch_xs):
            f.write(f"\n--- micro_batch {mi} ---\n")
            for b in range(x_cpu.shape[0]):
                ids = x_cpu[b].tolist()
                text = tokenizer.decode(ids)
                f.write(f"[sample {b}]\n{text}\n")


def train(args):
    assert 1 <= args.seq_len <= args.max_sequence_length
    assert args.grad_accum_steps > 0
    assert args.num_iterations >= 0
    assert args.wandb_log_interval > 0
    assert args.val_every >= 0
    assert args.eval_every >= 0
    assert args.checkpoint_every >= 0
    assert args.decay_steps >= 0

    local_rank_env = os.environ.get("LOCAL_RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    is_distributed = local_rank_env is not None and world_size_env is not None

    if is_distributed:
        init_process_group()
        rank = int(local_rank_env)
        world_size = int(world_size_env)
    else:
        rank = 0
        world_size = 1

    # rng / reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        device_type = "cuda"
    else:
        device = "cpu"
        device_type = "cpu"

    wandb_run = None
    should_log_wandb = args.wandb and rank == 0
    if should_log_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Weights & Biases is enabled but not installed. Install with `pip install wandb`."
            ) from exc
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )
    raw_run_name = wandb_run.name if wandb_run is not None else args.wandb_run_name
    checkpoint_run_name = None
    if raw_run_name:
        checkpoint_run_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_run_name).strip(
            "._-"
        )
        if not checkpoint_run_name:
            checkpoint_run_name = "run"

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    model = Model(
        ModelConfig(
            d_model=args.d_model,
            n_kv_heads=args.n_kv_heads,
            n_q_heads=args.n_q_heads,
            n_layers=args.n_layers,
            max_sequence_length=args.max_sequence_length,
            vocab_size=args.vocab_size,
            position_embedding_type=args.position_embedding_type,
        )
    )
    model.train()
    model.to(device)

    tokenizer = tiktoken.get_encoding("gpt2")
    eot_token_id = tokenizer.eot_token
    if not args.no_compile:
        model = torch.compile(model)

    if is_distributed:
        if device_type == "cuda":
            model = DDP(model, device_ids=[rank])
        else:
            model = DDP(model)

    # load dataset and stream-tokenize into packed batches
    train_dataset = get_dataset()
    if world_size > 1:
        train_dataset = train_dataset.shard(num_shards=world_size, index=rank)
    train_loader = DataLoader(
        args.batch_size, args.seq_len, train_dataset, args.data_buffer_size
    )
    val_loader = None
    if args.val_every > 0:
        val_dataset = get_dataset().skip(10_000_000)
        val_loader = DataLoader(
            args.batch_size, args.seq_len, val_dataset, args.data_buffer_size
        )

    # init the optimizer
    model_for_optim = model.module if is_distributed else model
    optimizer = model_for_optim.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        zero_stage=0,
    )

    min_lr = args.learning_rate * args.learning_rate_decay_frac

    timings = []
    norm = -1.0  # dummy value to print in inference-only mode
    loss_history: deque[float] = deque(maxlen=ROLLING_METRIC_WINDOW)
    norm_history: deque[float] = deque(maxlen=ROLLING_METRIC_WINDOW)
    smooth_loss = None
    smooth_norm = None
    smooth_beta = 0.95
    autocast_context = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    try:
        with autocast_context:
            for step in range(args.num_iterations + 1):
                t0 = time.time()
                # --------------- TRAINING SECTION BEGIN -----------------
                model.train()
                optimizer.zero_grad(set_to_none=True)
                # micro-batch loop where we do gradient accumulation to reach desired total batch size
                lossf = 0.0  # for getting the mean loss (as simple float) over the accumulation steps
                total_toks = 0
                batch_xs_for_log: list = []
                clone_batches_for_log = rank == 0 and len(loss_history) == ROLLING_METRIC_WINDOW
                for micro_step in range(args.grad_accum_steps):
                    # fetch a batch
                    x, y = next(train_loader)
                    if clone_batches_for_log:
                        batch_xs_for_log.append(x.detach().cpu().clone())

                    x, y = x.to(device), y.to(device)
                    attn_mask = None
                    if args.intra_doc_mask:
                        attn_mask = make_intra_document_attn_mask(x, eot_token_id)
                    # forward pass
                    _, loss = model(x, y, attn_mask=attn_mask)
                    total_toks += x.size(0) * x.size(1)
                    # we have to scale the loss to account for gradient accumulation,
                    # because the gradients just add on each successive backward().
                    # addition of gradients corresponds to a SUM in the objective, but
                    # instead of a SUM we want MEAN, so we scale the loss here
                    loss = loss / args.grad_accum_steps
                    lossf += loss.detach()  # keep track of the mean loss
                    # backward pass
                    loss.backward()
                lossf = lossf.item()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                norm_f = float(norm)
                smooth_loss = (
                    lossf
                    if smooth_loss is None
                    else smooth_beta * smooth_loss + (1.0 - smooth_beta) * lossf
                )
                smooth_norm = (
                    norm_f
                    if smooth_norm is None
                    else smooth_beta * smooth_norm + (1.0 - smooth_beta) * norm_f
                )
                if rank == 0:
                    loss_out = _is_outside_rolling_window(
                        smooth_loss, loss_history, frac=ROLLING_OUTLIER_FRAC
                    )
                    norm_out = _is_outside_rolling_window(
                        smooth_norm, norm_history, frac=ROLLING_OUTLIER_FRAC
                    )
                    if loss_out or norm_out:
                        _log_outlier_batches(
                            OUTLIER_LOG_PATH,
                            step_display=step + 1,
                            lossf=lossf,
                            grad_norm=norm_f,
                            tokenizer=tokenizer,
                            batch_xs=batch_xs_for_log,
                        )
                    loss_history.append(smooth_loss)
                    norm_history.append(smooth_norm)
                # determine and set the learning rate for this iteration
                lr = get_wsd_lr(
                    step,
                    learning_rate=args.learning_rate,
                    min_lr=min_lr,
                    warmup_iters=args.warmup_iters,
                    num_iterations=args.num_iterations,
                    decay_steps=args.decay_steps,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr
                # step the optimizer
                optimizer.step()
                # --------------- TRAINING SECTION END -------------------
                # everything that follows now is just diagnostics, prints, logging, etc.

                # time and print
                t1 = time.time()
                # the 0th iteration is often an outlier (much slower) => skip logging it
                if rank == 0 and step > 0:
                    print(
                        f"step {step + 1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1 - t0) * 1000:.2f} ms) | toks/s {total_toks / (t1 - t0):.2f}"
                    )
                    if wandb_run is not None and (
                        (step + 1) % args.wandb_log_interval == 0
                    ):
                        wandb_run.log(
                            {
                                "train/loss": lossf,
                                "train/norm": float(norm),
                                "train/lr": lr,
                                "train/toks_per_s": total_toks / (t1 - t0),
                            },
                            step=step + 1,
                        )

                if (
                    val_loader is not None
                    and args.val_every > 0
                    and ((step + 1) % args.val_every == 0)
                ):
                    model.eval()
                    val_loader.reset()
                    val_loss_sum = 0.0
                    with torch.no_grad():
                        for _ in range(args.grad_accum_steps):
                            xv, yv = next(val_loader)
                            xv, yv = xv.to(device), yv.to(device)
                            attn_mask = None
                            if args.intra_doc_mask:
                                attn_mask = make_intra_document_attn_mask(
                                    xv, eot_token_id
                                )
                            _, val_loss = model(xv, yv, attn_mask=attn_mask)
                            val_loss_sum += float(val_loss.detach())
                    val_loss_mean = val_loss_sum / args.grad_accum_steps

                    if is_distributed:
                        val_loss_tensor = torch.tensor(val_loss_mean, device=device)
                        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                        val_loss_mean = float(val_loss_tensor.item())

                    if rank == 0:
                        print(
                            f"step {step + 1:4d}/{args.num_iterations} | val loss {val_loss_mean:.6f}"
                        )
                        if wandb_run is not None:
                            wandb_run.log({"val/loss": val_loss_mean}, step=step + 1)

                if args.eval_every > 0 and ((step + 1) % args.eval_every == 0):
                    lm_eval.simple_evaluate(
                        model=model,
                        tasks=["mmlu"],
                    )

                _save_checkpoint_if_needed(
                    step=step,
                    args=args,
                    rank=rank,
                    checkpoint_run_name=checkpoint_run_name,
                    model=model,
                    is_distributed=is_distributed,
                    optimizer=optimizer,
                )

                # keep track of smooth timings, last 20 iterations
                if step > 0 and step > args.num_iterations - 20:
                    timings.append(t1 - t0)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if is_distributed:
            destroy_process_group()


if __name__ == "__main__":
    train(parse_args())
