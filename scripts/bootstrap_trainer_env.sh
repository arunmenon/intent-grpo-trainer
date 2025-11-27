#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a local venv and install core trainer deps.
# Torch is intentionally not pinned/installed here because CUDA builds are environment-specific;
# install the correct torch wheel for your GPU/driver after this script.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")"/.. && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found at ${PYTHON_BIN}. Set PYTHON_BIN to a valid interpreter." >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  echo "[bootstrap] creating venv at $VENV"
  "$PYTHON_BIN" -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "[bootstrap] upgrading pip"
python -m pip install --upgrade pip

echo "[bootstrap] installing core deps (transformers, trl, datasets, litellm)"
python -m pip install "transformers" "trl" "datasets" "litellm"

echo "[bootstrap] checking for torch"
if python - <<'PY' 2>/dev/null; then
import torch  # noqa: F401
print("torch already installed")
PY
then
  :
else
  cat <<'WARN'
[bootstrap] torch not installed.
Install a GPU-appropriate build, e.g.:
  pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
WARN
fi

cat <<DONE
[bootstrap] complete.
Activate the venv with:
  source $VENV/bin/activate

Run training via:
  python scripts/run_decision_grpo.py --model-id <model> --dataset examples/mock_rl_dataset.jsonl --steps 50 --batch-size 1 --grad-accum 1 --reward-variant decision_object
DONE
