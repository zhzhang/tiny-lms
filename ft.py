import argparse
import json
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a LoRA SFT experiment with TRL's SFTTrainer."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3.5-2B-Base",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="allenai/Dolci-Instruct-SFT",
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to load before making a local eval split.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/qwen3.5-2b-base-dolci-sft-lora",
        help="Where to write checkpoints and the final adapter.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Per-device eval batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of epochs when max_steps is not set.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging frequency in optimizer steps.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Evaluation frequency in optimizer steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--eval-size",
        type=int,
        default=2000,
        help="Number of held-out examples for evaluation. Use 0 to disable eval.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train examples for quick experiments.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on eval examples.",
    )
    parser.add_argument(
        "--dataset-num-proc",
        type=int,
        default=4,
        help="Processes to use for dataset normalization.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit loading for lower VRAM at the cost of extra complexity.",
    )
    parser.add_argument(
        "--packing",
        action="store_true",
        help="Enable example packing for better throughput.",
    )
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute loss only on assistant turns.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use gradient checkpointing to reduce activation memory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code to Hugging Face model/tokenizer loading.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, report metrics to Weights & Biases using this project name.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Resume from a previous Trainer checkpoint.",
    )
    return parser.parse_args()


def pick_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=True))
                continue
            parts.append(str(item))
        return "".join(parts)
    return str(content)


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    cleaned_messages = []
    for message in example["messages"]:
        role = message.get("role")
        content = normalize_content(message.get("content"))
        if role is None:
            continue
        cleaned_messages.append({"role": role, "content": content})
    return {"messages": cleaned_messages}


def is_trainable_example(example: dict[str, Any]) -> bool:
    messages = example["messages"]
    if len(messages) < 2:
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return bool(messages[-1]["content"].strip())


def maybe_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def load_and_prepare_datasets(
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset | None]:
    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
        streaming=True,
    )
    dataset = dataset.map(
        normalize_example,
        remove_columns=dataset.column_names,
        num_proc=args.dataset_num_proc,
        desc="Normalizing conversational messages",
    )
    dataset = dataset.filter(
        is_trainable_example,
        num_proc=args.dataset_num_proc,
        desc="Filtering valid SFT conversations",
    )

    if args.eval_size > 0:
        eval_size = min(args.eval_size, max(1, len(dataset) - 1))
        split_dataset = dataset.train_test_split(test_size=eval_size, seed=args.seed)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    train_dataset = maybe_limit(train_dataset, args.max_train_samples)
    if eval_dataset is not None:
        eval_dataset = maybe_limit(eval_dataset, args.max_eval_samples)

    return train_dataset, eval_dataset


def find_lora_target_modules(model: torch.nn.Module) -> list[str]:
    target_modules: set[str] = set()
    for module_name, module in model.named_modules():
        module_type_name = type(module).__name__
        is_supported_projection = isinstance(module, torch.nn.Linear) or (
            module_type_name in {"Conv1D", "Linear4bit", "Linear8bitLt"}
            and hasattr(module, "weight")
        )
        if not is_supported_projection:
            continue
        leaf_name = module_name.split(".")[-1]
        if leaf_name == "lm_head":
            continue
        if "embed" in module_name or "embedding" in module_name:
            continue
        target_modules.add(leaf_name)

    preferred_order = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    ordered_targets = [name for name in preferred_order if name in target_modules]
    ordered_targets.extend(sorted(target_modules - set(ordered_targets)))
    if not ordered_targets:
        raise ValueError("Could not infer LoRA target modules from the loaded model.")
    return ordered_targets


def ensure_chat_template(tokenizer: Any) -> None:
    if getattr(tokenizer, "chat_template", None):
        return

    # This fallback keeps the dataset in conversational format for SFTTrainer
    # and marks assistant spans so assistant_only_loss can work.
    tokenizer.chat_template = """{% for message in messages %}{% if message['role'] == 'system' %}system
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'user' %}user
{{ message['content'] }}{{ eos_token }}
{% elif message['role'] == 'assistant' %}assistant
{% generation %}{{ message['content'] }}{{ eos_token }}{% endgeneration %}
{% else %}{{ message['role'] }}
{{ message['content'] }}{{ eos_token }}
{% endif %}{% endfor %}{% if add_generation_prompt %}assistant
{% endif %}"""


def infer_model_max_length(model: torch.nn.Module) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def load_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Any, list[str]]:
    dtype = pick_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    ensure_chat_template(tokenizer)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if args.load_in_4bit:
        if not torch.cuda.is_available():
            raise ValueError("--load-in-4bit requires CUDA.")
        quant_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=quant_dtype,
        )
        model_kwargs["device_map"] = {"": 0}

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
            )
        elif hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = find_lora_target_modules(model)
    return model, tokenizer, target_modules


def build_trainer(args: argparse.Namespace) -> SFTTrainer:
    train_dataset, eval_dataset = load_and_prepare_datasets(args)
    model, tokenizer, target_modules = load_model_and_tokenizer(args)
    effective_max_length = args.max_seq_length
    model_max_length = infer_model_max_length(model)
    if model_max_length is not None:
        effective_max_length = min(effective_max_length, model_max_length)

    report_to = "wandb" if args.wandb_project else "none"
    if args.wandb_project:
        import os

        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    dtype = pick_dtype()
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_length=effective_max_length,
        packing=args.packing,
        assistant_only_loss=args.assistant_only_loss,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="no",
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        dataloader_num_workers=2,
        dataset_num_proc=args.dataset_num_proc,
        report_to=report_to,
        run_name=args.wandb_run_name,
        seed=args.seed,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        remove_unused_columns=False,
        eos_token=tokenizer.eos_token,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )

    print(f"Train examples: {len(train_dataset):,}")
    if eval_dataset is not None:
        print(f"Eval examples:  {len(eval_dataset):,}")
    print(f"Max sequence length: {effective_max_length}")
    print(f"LoRA target modules: {target_modules}")

    return SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    trainer = build_trainer(args)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.processing_class.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
