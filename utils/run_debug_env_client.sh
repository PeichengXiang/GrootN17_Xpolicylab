#!/bin/bash
set -euo pipefail

if [[ $# -lt 12 ]]; then
    echo "usage: $0 <eval_batch> <eval_env> <port> <bench_name> <task_name> <env_cfg_type> <policy_name> <additional_info> <benchmark_root> <seed> <env_gpu_id> <host> [protocol]" >&2
    exit 2
fi

eval_batch="${1}"
eval_env_conda_env="${2}"
free_port="${3}"
bench_name="${4}"
task_name="${5}"
env_cfg_type="${6}"
policy_name="${7}"
additional_info="${8}"
root_dir="${9}"
seed="${10}"
env_gpu_id="${11}"
policy_server_ip="${12:-localhost}"
protocol="${13:-ws}"

export PATH="/root/miniconda3/bin:/personal/miniconda3/bin:${PATH}"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
    echo "[CLIENT][ERROR] conda is required for the eval environment" >&2
    exit 1
fi
CONDA_BASE="$(${CONDA_BIN} info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda deactivate || true
conda activate "${eval_env_conda_env}"

echo -e "\033[34m[CLIENT] Activating Conda environment: ${eval_env_conda_env}\033[0m"
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${free_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

# In the external-workspace layout the benchmark checkout contains a pinned
# XPolicyLab copy, while the policy server is running the adapted checkout
# passed as EVAL_XPOLICY_ROOT.  Use the same checkout for the debug client's
# deploy import so the smoke test cannot silently exercise a stale adapter.
debug_root="${EVAL_XPOLICY_ROOT:-}"
if [[ -n "${debug_root}" && -f "${debug_root}/debug_env_client.py" ]]; then
    debug_client="${debug_root}/debug_env_client.py"
elif [[ -f "${root_dir}/XPolicyLab/debug_env_client.py" ]]; then
    debug_root="${root_dir}/XPolicyLab"
    debug_client="${debug_root}/debug_env_client.py"
elif [[ -f "${root_dir}/debug_env_client.py" ]]; then
    debug_root="${root_dir}"
    debug_client="${debug_root}/debug_env_client.py"
else
    echo "[CLIENT][ERROR] Could not locate debug_env_client.py" >&2
    exit 1
fi

export PYTHONPATH="${debug_root}:${root_dir}:${root_dir}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}"

client_python="${CONDA_PREFIX:-}/bin/python3"
if [[ ! -x "${client_python}" ]]; then
    client_python="$(command -v python3 || command -v python || true)"
fi
if [[ -z "${client_python}" ]]; then
    echo "[CLIENT][ERROR] Could not locate Python 3 in eval environment" >&2
    exit 1
fi

"${client_python}" "${debug_client}" \
    --bench_name "${bench_name}" \
    --task_name "${task_name}" \
    --env_cfg_type "${env_cfg_type}" \
    --policy_name "${policy_name}" \
    --protocol "${protocol}" \
    --host "${policy_server_ip}" \
    --port "${free_port}" \
    --eval_batch "${eval_batch}" \
    --obs_encoded "${DEBUG_OBS_ENCODED:-0}" \
    --eval_episode_num "${DEBUG_EVAL_EPISODES:-10}" \
    --episode_step_limit "${DEBUG_EPISODE_STEPS:-20}"
