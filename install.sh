#!/usr/bin/env sh
set -eu

REPO_URL="${REPO_URL:-https://github.com/zhzhang/tiny-lms.git}"
TARGET_DIR="${TARGET_DIR:-gpt2}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command git
require_command curl

if [ -e "$TARGET_DIR" ]; then
  echo "Target path already exists: $TARGET_DIR" >&2
  echo "Set TARGET_DIR to a different folder and rerun." >&2
  exit 1
fi

echo "Cloning repository into $TARGET_DIR..."
git clone --depth 1 "$REPO_URL" "$TARGET_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
elif [ -x "$HOME/.cargo/bin/uv" ]; then
  UV_BIN="$HOME/.cargo/bin/uv"
else
  echo "uv installation completed but the binary was not found on PATH." >&2
  echo "Open a new shell or add ~/.local/bin to PATH, then run: uv sync" >&2
  exit 1
fi

echo "Installing project dependencies with uv..."
cd "$TARGET_DIR"
if [ -f "uv.lock" ]; then
  "$UV_BIN" sync --frozen
else
  "$UV_BIN" sync
fi

echo "Install complete."
echo "Next: cd $TARGET_DIR && uv run python train.py"
