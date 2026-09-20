#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"

echo "[GR00T_N17] GR00T_ROOT=${GR00T_ROOT}"
echo "[GR00T_N17] XPOLICYLAB_ROOT=${XPOLICYLAB_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

cd "${GR00T_ROOT}"
# NOTE: pyproject pins `[tool.uv] required-environments` to both x86_64 and aarch64,
# and the aarch64 torchcodec/flash-attn wheels under scripts/deployment/dgpu/wheels/
# are not shipped. `uv sync` therefore fails to lock on an x86_64 GPU host because it
# must resolve the (missing) aarch64 wheels. We instead create the venv and use
# `uv pip install -e .`, which resolves only for the *current* platform (x86_64) and
# honors [tool.uv.sources] / [[tool.uv.index]] while ignoring required-environments.
uv venv --clear --python 3.10
uv pip install -e .
.venv/bin/python -c "import gr00t; print('GR00T ok')"

uv pip install -e "${XPOLICYLAB_ROOT}"
uv pip install h5py pyyaml
# Only the lightweight LeRobot Python package is missing from the GR00T
# environment. Its runtime dependencies are already supplied by GR00T; avoid a
# second resolver pass that could replace the pinned torch/CUDA stack.
uv pip install --no-deps "lerobot==0.4.4"
.venv/bin/python -c "import av, h5py, lerobot, pyarrow, XPolicyLab; print('XPolicyLab/converter deps ok', lerobot.__version__)"
.venv/bin/python - "${GR00T_ROOT}" <<'PY'
import importlib.util
from pathlib import Path
import sys

converter_path = (
    Path(sys.argv[1]) / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"
)
spec = importlib.util.spec_from_file_location("groot_convert_v3_to_v2_smoke", converter_path)
if spec is None or spec.loader is None:
    raise ImportError(converter_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print("GR00T v3-to-v2 converter imports ok")
PY

echo "[GR00T_N17] Installation finished."
echo "[GR00T_N17] Policy server env: source ${GR00T_ROOT}/.venv/bin/activate"
echo "[GR00T_N17] Eval example:"
echo "  bash eval.sh RoboDojo sweep_blocks RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv mibot"
