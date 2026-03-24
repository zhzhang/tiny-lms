import argparse
import json

import lm_eval
import tiktoken
import torch
from lm_eval.api.model import LM

from model import Model, ModelConfig, PositionEmbeddingType


class LMEvalHarness(LM):
    def __init__(
        self,
        *,
        model: Model,
        tokenizer,
        device: str,
        max_length: int,
        max_gen_toks: int = 32,
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_length = max_length
        self.max_gen_toks = max_gen_toks
        self.prefix_token_id = tokenizer.eot_token

    @property
    def tokenizer_name(self) -> str:
        return "gpt2"

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode_ordinary(text)

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def _score_continuation(
        self, context_tokens: list[int], continuation_tokens: list[int]
    ) -> tuple[float, bool]:
        if not continuation_tokens:
            return 0.0, True

        prefix_tokens = context_tokens if context_tokens else [self.prefix_token_id]
        full_tokens = prefix_tokens + continuation_tokens
        max_full_tokens = self.max_length + 1
        overflow = max(len(full_tokens) - max_full_tokens, 0)
        if overflow:
            full_tokens = full_tokens[-max_full_tokens:]

        input_ids = full_tokens[:-1]
        target_ids = full_tokens[1:]
        score_start = max(len(prefix_tokens) - 1 - overflow, 0)

        x = torch.tensor([input_ids], device=self.device, dtype=torch.long)
        with torch.no_grad():
            logits, _ = self.model(x)
            log_probs = torch.log_softmax(logits[0], dim=-1)

        target_tensor = torch.tensor(
            target_ids[score_start:], device=self.device, dtype=torch.long
        )
        positions = torch.arange(score_start, len(target_ids), device=self.device)
        token_log_probs = log_probs[positions, target_tensor]
        greedy_tokens = logits[0, score_start:].argmax(dim=-1)
        is_greedy = bool(torch.equal(greedy_tokens, target_tensor))
        return float(token_log_probs.sum().item()), is_greedy

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        outputs: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            result = self._score_continuation(
                self._encode(context), self._encode(continuation)
            )
            self.cache_hook.add_partial("loglikelihood", request.args, result)
            outputs.append(result)
        return outputs

    def loglikelihood_rolling(self, requests) -> list[float]:
        outputs: list[float] = []
        for request in requests:
            (text,) = request.args
            text_tokens = self._encode(text)
            if not text_tokens:
                outputs.append(0.0)
                continue

            total_logprob = 0.0
            start = 0
            while start < len(text_tokens):
                end = min(start + self.max_length, len(text_tokens))
                prefix_start = max(0, start - self.max_length + 1)
                prefix_tokens = text_tokens[prefix_start:start]
                continuation_tokens = text_tokens[start:end]
                chunk_logprob, _ = self._score_continuation(
                    prefix_tokens, continuation_tokens
                )
                total_logprob += chunk_logprob
                start = end

            self.cache_hook.add_partial(
                "loglikelihood_rolling", request.args, total_logprob
            )
            outputs.append(total_logprob)
        return outputs

    def generate_until(self, requests) -> list[str]:
        outputs: list[str] = []
        for request in requests:
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", []) if isinstance(gen_kwargs, dict) else []
            if isinstance(until, str):
                until = [until]

            tokens = self._encode(context)
            if not tokens:
                tokens = [self.prefix_token_id]

            generated: list[int] = []
            for _ in range(self.max_gen_toks):
                window = tokens[-self.max_length :]
                x = torch.tensor([window], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    logits, _ = self.model(x)
                next_token = int(logits[0, -1].argmax().item())
                tokens.append(next_token)
                generated.append(next_token)
                generated_text = self._decode(generated)
                if any(stop and stop in generated_text for stop in until):
                    break

            generated_text = self._decode(generated)
            for stop in until:
                if stop and stop in generated_text:
                    generated_text = generated_text.split(stop)[0]
                    break
            self.cache_hook.add_partial("generate_until", request.args, generated_text)
            outputs.append(generated_text)
        return outputs


def evaluate_model(model: LMEvalHarness) -> dict:
    results = lm_eval.simple_evaluate(
        model=model,
        tasks=["hellaswag"],
        num_fewshot=0,
        limit=None,
        log_samples=False,
        bootstrap_iters=0,
    )
    output = {}
    output["eval/hellaswag-acc-norm"] = results["results"]["hellaswag"]["acc_norm,none"]
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Instantiate a scratch tiny-lms model and run lm_eval."
    )
    parser.add_argument("--task", default="hellaswag")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--n-q-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument(
        "--position-embedding-type",
        choices=[member.value for member in PositionEmbeddingType],
        default=PositionEmbeddingType.LEARNED.value,
    )
    parser.add_argument("--max-gen-toks", type=int, default=32)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> LMEvalHarness:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    config = ModelConfig(
        d_model=args.d_model,
        n_kv_heads=args.n_kv_heads,
        n_q_heads=args.n_q_heads,
        n_layers=args.n_layers,
        max_sequence_length=args.max_sequence_length,
        vocab_size=args.vocab_size,
        position_embedding_type=PositionEmbeddingType(args.position_embedding_type),
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    model = Model(config)
    return LMEvalHarness(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_length=args.max_sequence_length,
        max_gen_toks=args.max_gen_toks,
    )


def main() -> None:
    args = parse_args()
    limit = None if args.limit <= 0 else args.limit
    lm = build_model(args)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task],
        num_fewshot=0,
        limit=limit,
        log_samples=False,
        bootstrap_iters=0,
    )
    print(
        json.dumps(
            {
                "config": {
                    "task": args.task,
                    "limit": limit,
                    "device": args.device,
                    "d_model": args.d_model,
                    "n_layers": args.n_layers,
                    "max_sequence_length": args.max_sequence_length,
                    "position_embedding_type": args.position_embedding_type,
                },
                "results": results["results"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
