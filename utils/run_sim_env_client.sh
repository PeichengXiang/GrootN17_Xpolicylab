#!/bin/bash
set -euo pipefail

if [[ $# -lt 12 ]]; then
    echo "usage: $0 <eval_batch> <eval_env> <port> <bench_name> <task_name> <env_cfg_type> <policy_name> <additional_info> <benchmark_root> <seed> <env_gpu_id> <host> [protocol]" >&2
    exit 2
fi

eval_batch="${1}"
eval_env_conda_env="${2}"
policy_server_port="${3}"
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
echo -e "\033[34m[CLIENT] Connecting to server ${policy_server_ip}:${policy_server_port}...\033[0m"
echo -e "\033[34m[CLIENT] Watch for green [CONNECTED]; yellow [RECONNECT] means the client is retrying.\033[0m"

benchmark_root="${EVAL_MAIN_ROOT:-${root_dir}}"
eval_entry="${benchmark_root}/scripts/eval_policy.sh"
if [[ ! -x "${eval_entry}" ]]; then
    echo "[CLIENT][ERROR] Missing benchmark eval entrypoint: ${eval_entry}" >&2
    exit 1
fi

xpolicy_root="${EVAL_XPOLICY_ROOT:-${root_dir}/XPolicyLab}"
bridge_args=(
    --bench_name "${bench_name}"
    --task_name "${task_name}"
    --env_cfg_type "${env_cfg_type}"
    --policy_name "${policy_name}"
    --host "${policy_server_ip}"
    --port "${policy_server_port}"
    --protocol "${protocol}"
    --eval_batch "${eval_batch}"
    --root_dir "${benchmark_root}"
    --xpolicy-root "${xpolicy_root}"
    --device_id "${env_gpu_id}"
    --additional_info "${additional_info}"
    --seed "${seed}"
)
# The released benchmark is sometimes unpacked as a sibling checkout below
# the integration workspace.  Point the runtime at that real IsaacLab tree
# while retaining the top-level bridge for task/result bookkeeping.
if [[ -d "${benchmark_root}/Ego_Humanoid_Manipulation_Benchmark/source" ]]; then
    bridge_args+=(--egovla-root "${benchmark_root}/Ego_Humanoid_Manipulation_Benchmark")
fi
if [[ -d "${benchmark_root}/EgoVLA_Release" ]]; then
    bridge_args+=(--egovla-release-root "${benchmark_root}/EgoVLA_Release")
fi
if [[ -n "${EGOVLA_CHECKPOINT:-}" ]]; then
    bridge_args+=(--checkpoint "${EGOVLA_CHECKPOINT}")
fi
if [[ -n "${EGOVLA_ACTION_TYPE:-}" ]]; then
    bridge_args+=(--action-type "${EGOVLA_ACTION_TYPE}")
fi
EVAL_XPOLICY_ROOT="${xpolicy_root}" \
EVAL_MAIN_ROOT="${benchmark_root}" \
PYTHONPATH="${xpolicy_root}:${root_dir}:${benchmark_root}:${benchmark_root}/XPolicyLab${PYTHONPATH:+:${PYTHONPATH}}" \
bash "${eval_entry}" "${bridge_args[@]}"
