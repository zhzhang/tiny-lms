import argparse
import glob
import math
import os
import re
import time
from contextlib import nullcontext

import numpy as np
import tiktoken
import torch
from hellaswag import iterate_examples, render_example
from model import Model, ModelConfig, PositionEmbeddingType
from torch import distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

torch.autograd.set_detect_anomaly(True)


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


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
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--learning-rate-decay-frac", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-iters", type=int, default=0)
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
    return same_doc & causal


@torch.no_grad()
def evaluate_hellaswag(model, device, rank, world_size):
    num_correct_norm = 0
    num_correct = 0
    num_total = 0

    for idx, example in enumerate(iterate_examples("val")):
        # Split eval examples across processes so all ranks do useful work.
        if idx % world_size != rank:
            continue
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        logits, _ = model(tokens)
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(
            flat_shift_logits, flat_shift_tokens, reduction="none"
        )
        shift_losses = shift_losses.view(tokens.size(0), -1)

        shift_mask = mask[..., 1:].contiguous()
        masked_shift_losses = shift_losses * shift_mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    counts = torch.tensor(
        [num_correct, num_correct_norm, num_total], device=device, dtype=torch.long
    )
    if world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    total = int(counts[2].item())
    acc = float(counts[0].item()) / total
    acc_norm = float(counts[1].item()) / total
    return acc, acc_norm


def _run_hellaswag_eval_if_needed(
    *,
    step,
    args,
    model,
    device,
    rank,
    world_size,
    wandb_run,
):
    if args.eval_every <= 0 or (step + 1) % args.eval_every != 0:
        return

    model.eval()
    eval_t0 = time.time()
    hs_acc, hs_acc_norm = evaluate_hellaswag(
        model=model,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    eval_t1 = time.time()

    if rank == 0:
        print(
            f"step {step + 1:4d}/{args.num_iterations} | hellaswag acc {hs_acc:.4f} | hellaswag acc_norm {hs_acc_norm:.4f} | ({(eval_t1 - eval_t0):.2f} s)"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "hellaswag/acc": hs_acc,
                    "hellaswag/acc_norm": hs_acc_norm,
                },
                step=step + 1,
            )


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


def train(args):
    assert 1 <= args.seq_len <= args.max_sequence_length
    assert args.grad_accum_steps > 0
    assert args.num_iterations >= 0
    assert args.wandb_log_interval > 0
    assert args.val_every >= 0
    assert args.eval_every >= 0
    assert args.checkpoint_every >= 0

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

    # load tokens
    train_loader = DistributedDataLoader(
        args.train_bin_pattern, args.batch_size, args.seq_len, rank, world_size
    )
    val_loader = None
    if args.val_every > 0:
        val_loader = DistributedDataLoader(
            args.val_bin_pattern, args.batch_size, args.seq_len, rank, world_size
        )

    # init the optimizer
    model_for_optim = model.module if is_distributed else model
    optimizer = model_for_optim.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        zero_stage=0,
    )

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        if args.warmup_iters > 0 and it < args.warmup_iters:
            return args.learning_rate * (it + 1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.num_iterations:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        denom = max(args.num_iterations - args.warmup_iters, 1)
        decay_ratio = (it - args.warmup_iters) / denom
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (
            1.0 + math.cos(math.pi * decay_ratio)
        )  # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.learning_rate - min_lr)

    timings = []
    norm = -1.0  # dummy value to print in inference-only mode
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
                for micro_step in range(args.grad_accum_steps):
                    # fetch a batch
                    x, y = train_loader.next_batch()

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
                # determine and set the learning rate for this iteration
                lr = get_lr(step)
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
                            xv, yv = val_loader.next_batch()
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

                _run_hellaswag_eval_if_needed(
                    step=step,
                    args=args,
                    model=model,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    wandb_run=wandb_run,
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
