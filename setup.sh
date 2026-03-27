#!/usr/bin/env bash
# Colab-friendly environment bootstrap.
# Colab ships PyTorch pinned to its CUDA runtime — verify with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`.
# Only force-reinstall torch if you hit ABI mismatches when building mamba-ssm / Triton extensions.
set -euo pipefail

pip install -q --upgrade pip
pip install -q ninja packaging wheel

# Triton: install version compatible with the active torch+CUDA combo (may already exist on Colab).
python - <<'PY' || true
import importlib.util
if importlib.util.find_spec("triton") is None:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "triton"])
PY

# Core Python deps (adds mamba-ssm; compilation can take several minutes on first Colab run).
pip install -q -r requirements_colab.txt

echo "setup.sh finished. If mamba-ssm failed to build, the encoder falls back to a lightweight mixer."
