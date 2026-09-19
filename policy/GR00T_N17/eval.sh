#!/bin/bash
set -euo pipefail

if [[ $# -lt 10 ]]; then
    echo "usage: $0 <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <policy_gpu_id> <env_gpu_id> <policy_env> <eval_env>" >&2
    exit 2
fi

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
env_gpu_id=$8
policy_conda_env=$9
eval_env_conda_env=${10}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

# The EgoVLA bridge passes the benchmark workspace as EGOVLA_WORKSPACE_ROOT
# when the policy checkout is kept outside the benchmark repository.  Preserve
# an explicit EVAL_MAIN_ROOT override for older/direct invocations.
if [[ -z "${EVAL_MAIN_ROOT:-}" && -n "${EGOVLA_WORKSPACE_ROOT:-}" ]]; then
    export EVAL_MAIN_ROOT="${EGOVLA_WORKSPACE_ROOT}"
fi
export EVAL_XPOLICY_ROOT="${XPL_ROOT}"

SERVER_SCRIPT="${SCRIPT_DIR}/setup_eval_policy_server.sh"
CLIENT_SCRIPT="${SCRIPT_DIR}/setup_eval_env_client.sh"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
policy_server_ip="localhost"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "[MAIN] kill server ${SERVER_PID}"
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[MAIN] start server, policy_server_port=${policy_server_port}"

bash "${SERVER_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${policy_gpu_id}" \
    "${policy_conda_env}" \
    "${policy_server_port}" &

SERVER_PID=$!

# Cold starts copy ~7GB shards off network into memfd (seen 300s+ on local)
# before the policy port opens.  600s is just short of that plus model load.
policy_server_wait_s="${GR00T_POLICY_SERVER_WAIT_S:-1800}"
bash "${UTILS_DIR}/wait_for_policy_server.sh" "${policy_server_ip}" "${policy_server_port}" "${SERVER_PID}" "Policy server" "${policy_server_wait_s}"

echo "[MAIN] start client, server=${policy_server_ip}:${policy_server_port}"

# The simulator resumes the latest incomplete result that has the same
# additional_info.  Keep Web jobs isolated from earlier cancelled jobs while
# preserving resume behavior when the same Web run itself is restarted.
additional_info="ckpt_name=${ckpt_name},action_type=${action_type}"
if [[ "${LUMINIS_RUN_ID:-}" == web-* ]]; then
    additional_info+=",eval_web_run=${LUMINIS_RUN_ID}"
fi

bash "${CLIENT_SCRIPT}" \
    "${bench_name}" \
    "${task_name}" \
    "${ckpt_name}" \
    "${env_cfg_type}" \
    "${action_type}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${eval_env_conda_env}" \
    "${additional_info}" \
    "${policy_server_port}" \
    "${policy_server_ip}"

echo "[MAIN] eval finished"
