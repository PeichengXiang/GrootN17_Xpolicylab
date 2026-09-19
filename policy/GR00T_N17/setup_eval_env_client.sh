#!/bin/bash
set -euo pipefail

if [[ $# -lt 11 ]]; then
    echo "usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <env_gpu_id> <eval_env> <additional_info> <port> <host>" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
env_gpu_id=$7
eval_env_conda_env=$8
additional_info=$9

# The shared setup_env_client parses deploy.yml before it activates the
# simulator environment.  Resolve YAML with the exact Web eval environment;
# host python3 is not guaranteed to carry PyYAML on remote nodes.
if [[ -z "${PYTHON_BIN:-}" || ! -x "${PYTHON_BIN}" ]]; then
    if [[ -x "${eval_env_conda_env}/bin/python" ]]; then
        export PYTHON_BIN="${eval_env_conda_env}/bin/python"
    fi
fi
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
BENCH_ROOT="${EVAL_MAIN_ROOT:-${EGOVLA_WORKSPACE_ROOT:-${DEFAULT_BENCH_ROOT}}}"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"
if [[ ! -f "${BENCH_ROOT}/scripts/eval_policy.sh" ]]; then
    echo "[CLIENT][ERROR] Missing simulator entrypoint: ${BENCH_ROOT}/scripts/eval_policy.sh" >&2
    echo "[CLIENT][ERROR] Set EVAL_MAIN_ROOT to the simulator checkout." >&2
    exit 1
fi

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"
echo "[CLIENT] eval main root=${BENCH_ROOT}"

# Preserve the full checkpoint path/action type for the simulator bridge.  The
# legacy additional_info fallback is comma-delimited and cannot safely carry a
# path containing commas or equals signs.
export EGOVLA_CHECKPOINT="${ckpt_name}"
export EGOVLA_ACTION_TYPE="${action_type}"

EVAL_XPOLICY_ROOT="${XPL_ROOT}" \
EVAL_MAIN_ROOT="${BENCH_ROOT}" \
PYTHONPATH="${XPL_ROOT}:${XPL_ROOT}/..:${BENCH_ROOT}:${BENCH_ROOT}/XPolicyLab:${PYTHONPATH:-}" \
bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
