#!/bin/bash
set -euo pipefail

if [[ $# -lt 9 ]]; then
    echo "usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <policy_env> <port> [host]" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-${EGOVLA_WORKSPACE_ROOT:-${DEFAULT_BENCH_ROOT}}}"
GR00T_ROOT="${SCRIPT_DIR}/gr00t_n17"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

if [[ ! -f "${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" ]]; then
    echo "[SERVER][ERROR] Missing simulator env config: ${BENCH_ROOT}/env_cfg/${env_cfg_type}.yml" >&2
    echo "[SERVER][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi

if [[ -x "${UTILS_DIR}/get_action_dim.sh" ]]; then
    action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
else
    action_dim=$(python3 - "${XPL_ROOT}/utils/robot/_robot_info.json" "${env_cfg_type}" <<'PYDIM'
import json
import sys

robots = json.load(open(sys.argv[1], encoding="utf-8"))
info = robots[sys.argv[2]]
print(sum(info["arm_dim"]) + sum(info["ee_dim"]))
PYDIM
    )
fi

if [[ "${env_cfg_type}" == "ego_h1_inspire" && "${action_dim}" != "38" ]]; then
    echo "[SERVER][ERROR] ego_h1_inspire must expose 38 policy dimensions, got ${action_dim}" >&2
    exit 1
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] eval main root=${BENCH_ROOT}"

resolve_uv_env() {
    local raw_path=$1
    if [[ "${raw_path}" == "uv" ]]; then
        XPOLICY_DEPLOY_YAML="${yaml_file}" SCRIPT_DIR_FOR_ENV="${SCRIPT_DIR}" python3 - <<'PYENV'
import os
import yaml
from pathlib import Path
cfg = yaml.safe_load(open(os.environ["XPOLICY_DEPLOY_YAML"], encoding="utf-8")) or {}
path = Path(cfg.get("policy_uv_env_path", "gr00t_n17")).expanduser()
if not path.is_absolute():
    path = (Path(os.environ["SCRIPT_DIR_FOR_ENV"]) / path).resolve()
print(path)
PYENV
    else
        SCRIPT_DIR_FOR_ENV="${SCRIPT_DIR}" RAW_ENV_PATH="${raw_path}" python3 - <<'PYENV'
import os
from pathlib import Path
path = Path(os.environ["RAW_ENV_PATH"]).expanduser()
if not path.is_absolute():
    path = (Path(os.environ["SCRIPT_DIR_FOR_ENV"]) / path).resolve()
print(path)
PYENV
    fi
}

if [[ "${policy_conda_env}" == "uv" || "${policy_conda_env}" == */* ]]; then
    policy_uv_env_path="$(resolve_uv_env "${policy_conda_env}")"
    # Accept a uv project, its .venv, or a direct Python/prefix path.  The
    # GrootN17 checkout uses a project-root argument, while the bridge may pass
    # the already-created .venv prefix.
    if [[ -x "${policy_uv_env_path}" && "$(basename "${policy_uv_env_path}")" == python* ]]; then
        PYTHON_BIN="${policy_uv_env_path}"
    elif [[ -x "${policy_uv_env_path}/bin/python" ]]; then
        PYTHON_BIN="${policy_uv_env_path}/bin/python"
    elif [[ -x "${policy_uv_env_path}/.venv/bin/python" ]]; then
        PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
    else
        PYTHON_BIN="${policy_uv_env_path}/.venv/bin/python"
    fi
    echo "[SERVER] Using uv environment: ${policy_uv_env_path}"
else
    export PATH="/root/miniconda3/bin:/personal/miniconda3/bin:${PATH}"
    CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
    if [[ -z "${CONDA_BIN}" ]]; then
        echo "[SERVER][ERROR] conda is required for named policy environments" >&2
        exit 1
    fi
    source "$(${CONDA_BIN} info --base)/etc/profile.d/conda.sh"
    echo "[SERVER] Activating Conda environment: ${policy_conda_env}"
    conda activate "${policy_conda_env}"
    PYTHON_BIN="$(command -v python)"
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python not found: ${PYTHON_BIN}" >&2
    exit 1
fi

export PATH="/personal/miniconda3/envs/Luminis/bin:${PATH}"
export PYTHONPATH="${XPL_ROOT}:${GR00T_ROOT}:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}"
# Allow HuggingFace download for cosmos_model_path; set HF_HUB_OFFLINE=1 for fully offline deploy
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export GR00T_VIDEO_BACKEND="${GR00T_VIDEO_BACKEND:-pyav}"
export GR00T_TRUST_VIDEO_LENGTHS="${GR00T_TRUST_VIDEO_LENGTHS:-1}"
if [[ -z "${GR00T_COSMOS_MODEL:-}" && -d "${XPL_ROOT}/pretrain_model/Cosmos-Reason2-2B" ]]; then
    export GR00T_COSMOS_MODEL="${XPL_ROOT}/pretrain_model/Cosmos-Reason2-2B"
fi

if [[ "${policy_gpu_id}" == "cpu" || "${policy_gpu_id}" == "none" || "${policy_gpu_id}" == "-1" ]]; then
    # Useful for a low-impact model/server smoke while the training GPUs are
    # occupied.  Model.__init__ detects CUDA as unavailable and selects CPU.
    CUDA_DEVICE_VALUE=""
else
    CUDA_DEVICE_VALUE="${policy_gpu_id}"
fi

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_VALUE}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
