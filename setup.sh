#!/usr/bin/env bash
# Colab-friendly environment bootstrap.
# Colab ships PyTorch + CUDA preinstalled — do not reinstall torch unless you know you need a pin.
#
# Default: no mamba-ssm (avoids noisy CUDA wheel build failures). The encoder uses a fast
# GLU/Linear sequence mixer instead; see models/encoder_vim.py.
#
# To try real Mamba (may still fail on some runtimes):
#   INSTALL_MAMBA=1 bash setup.sh
set -euo pipefail

pip install -q --upgrade pip
pip install -q ninja packaging wheel

# Triton: only install if missing (avoid clobbering Colab’s matched build).
python - <<'PY' || true
import importlib.util, subprocess, sys
if importlib.util.find_spec("triton") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "triton"])
PY

pip install -q -r requirements_colab.txt

if [[ "${INSTALL_MAMBA:-0}" == "1" ]]; then
  echo "INSTALL_MAMBA=1: attempting causal-conv1d + mamba-ssm (CUDA compile; may fail)..."
  set +e
  pip install -q --no-cache-dir --no-build-isolation -r requirements_colab_mamba.txt
  status=$?
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "WARNING: mamba-ssm build failed (exit $status). Continuing with encoder fallback (no Mamba)."
  fi
fi

python - <<'PY'
try:
    import mamba_ssm  # noqa: F401
    print("encoder_vim: mamba_ssm import OK — ViM will use Mamba blocks.")
except ImportError:
    print("encoder_vim: mamba_ssm not installed — ViM will use lightweight mixer blocks.")
PY

echo "setup.sh finished."
