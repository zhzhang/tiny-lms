#!/usr/bin/env python3
"""
Fetch FineWeb-Edu (sample-10BT) from Hugging Face and report compression ratio,
fertility, and PCW for a chosen tokenizer. Default: gpt2 (tiktoken).
"""

import argparse
from datasets import load_dataset


# Dataset: (hf_id, config_name, text_column)
FINEWEB_EDU = ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text")


def get_tokenizer(name: str):
    if name == "gpt2":
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return lambda s: enc.encode_ordinary(s)
    if name.startswith("gpt2"):
        import tiktoken
        enc = tiktoken.get_encoding(name)
        return lambda s: enc.encode_ordinary(s)
    # Hugging Face tokenizers
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    return lambda s: tok.encode(s, add_special_tokens=False)


def measure_compression(dataset_id: str, config_name: str | None, text_col: str, tokenize, max_bytes: int):
    kwargs = {"path": dataset_id, "split": "train", "streaming": True}
    if config_name:
        kwargs["name"] = config_name
    ds = load_dataset(**kwargs)

    total_bytes = 0
    total_tokens = 0
    total_words = 0
    continued_words = 0  # words that tokenize to more than 1 token

    for row in ds:
        text = row.get(text_col) or ""
        if not isinstance(text, str):
            continue
        raw = text.encode("utf-8")
        total_bytes += len(raw)
        tokens = tokenize(text)
        total_tokens += len(tokens)

        # Word-level stats: tokenize each word with leading space (in-context)
        for word in text.split():
            if not word:
                continue
            n_tokens = len(tokenize(" " + word))
            total_words += 1
            if n_tokens > 1:
                continued_words += 1

        if total_bytes >= max_bytes:
            break

    if total_tokens == 0:
        return None, None
    ratio = total_bytes / total_tokens
    fertility = total_tokens / total_words if total_words else 0.0
    pcw = continued_words / total_words if total_words else 0.0
    return ratio, (total_bytes, total_tokens, total_words, continued_words, fertility, pcw)


def main():
    ap = argparse.ArgumentParser(description="Measure tokenizer compression on FineWeb-Edu (sample-10BT)")
    ap.add_argument("--tokenizer", default="gpt2", help="Tokenizer: gpt2 (default), gpt2-50k, or HF model id")
    ap.add_argument("--max-bytes", type=int, default=10_000_000, help="Max bytes of text to sample (default 10M)")
    args = ap.parse_args()

    tokenize = get_tokenizer(args.tokenizer)
    datasets_to_run = [("FineWeb-Edu", *FINEWEB_EDU)]

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Max bytes to sample: {args.max_bytes:,}")
    print()

    for label, dataset_id, config_name, text_col in datasets_to_run:
        print(f"Loading {label} ({dataset_id})...")
        try:
            ratio, counts = measure_compression(
                dataset_id, config_name, text_col, tokenize, args.max_bytes
            )
        except Exception as e:
            print(f"  Error: {e}")
            continue
        if ratio is None:
            print(f"  No text rows.")
            continue
        total_bytes, total_tokens, total_words, continued_words, fertility, pcw = counts
        print(f"  Bytes: {total_bytes:,}  Tokens: {total_tokens:,}  Words: {total_words:,}")
        print(f"  Compression ratio (bytes/token): {ratio:.2f}")
        print(f"  Fertility (tokens/word): {fertility:.3f}")
        print(f"  PCW (proportion of continued words, >1 token): {pcw:.2%}")
        print()


if __name__ == "__main__":
    main()
