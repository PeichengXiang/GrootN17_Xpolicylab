#!/usr/bin/env bash
set -euo pipefail

ckpt_name=${1:-groot_n17_lerobotV21_joint54}
seed=${2:-0}
gpu_id=${3:-0,1,2,3,4,5,6,7}

bench_name=Spark0_bench
env_cfg_type=tianji_marvin_wuji
action_type=joint

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
GR00T_PYTHON="${GR00T_ROOT}/.venv/bin/python"
DATA_ROOT="${GR00T_LEROBOT_HOME:-}"

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set GR00T_LEROBOT_HOME to the LeRobot datasets root." >&2
  exit 1
fi
for identifier in "${ckpt_name}" "${seed}"; do
  if [[ ! "${identifier}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Unsafe run identifier: '${identifier}'. Use only letters, digits, '_' and '-'." >&2
    exit 1
  fi
done

IFS=',' read -r -a gpu_ids <<< "${gpu_id}"
if (( ${#gpu_ids[@]} != 8 )); then
  echo "Spark0 joint54 recipe requires exactly 8 GPU ids, got '${gpu_id}'." >&2
  exit 1
fi
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]] || [[ -n "${seen_gpus[${gpu}]:-}" ]]; then
    echo "GPU ids must be eight unique non-negative integers, got '${gpu_id}'." >&2
    exit 1
  fi
  seen_gpus[${gpu}]=1
done

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
ckpt_setting="${data_setting}-${seed}"
dataset_path="${DATA_ROOT}/${data_setting}"
output_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
modality_config="${POLICY_DIR}/configs/${env_cfg_type}_config.py"
dataset_auditor="${POLICY_DIR}/scripts/audit_spark0_groot_v21.py"
base_model="${GR00T_BASE_MODEL:-${XPL_ROOT}/pretrain_model/GR00T-N1.7-3B}"
cosmos_model="${GR00T_COSMOS_MODEL:-${XPL_ROOT}/pretrain_model/Cosmos-Reason2-2B}"

if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  echo "Output already exists; refusing implicit resume: ${output_dir}" >&2
  exit 1
fi
case "${base_model}" in
  checkpoints/*|*/policy/GR00T_N17/checkpoints/*)
    echo "Refusing a fine-tuned GR00T checkpoint as the Spark0 base model: ${base_model}" >&2
    exit 1
    ;;
esac
if [[ -e "${base_model}" ]]; then
  resolved_base="$(realpath -e "${base_model}")"
  resolved_checkpoints="$(realpath -m "${POLICY_DIR}/checkpoints")"
  case "${resolved_base}" in
    "${resolved_checkpoints}"|"${resolved_checkpoints}"/*)
      echo "Refusing a fine-tuned GR00T checkpoint as the Spark0 base model: ${resolved_base}" >&2
      exit 1
      ;;
  esac
fi
if [[ ! -f "${base_model}/config.json" ]]; then
  echo "Local GR00T base model is incomplete: ${base_model}" >&2
  exit 1
fi
if [[ ! -f "${cosmos_model}/config.json" ]]; then
  echo "Local Cosmos model is incomplete: ${cosmos_model}" >&2
  exit 1
fi

export PYTHONPATH="${XPL_ROOT}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${GR00T_PYTHON}" - "${XPL_ROOT}" "${GR00T_ROOT}" <<'PY'
from pathlib import Path
import sys

import XPolicyLab.utils.process_data as process_data
import gr00t

xpl_root = Path(sys.argv[1]).resolve()
gr00t_root = Path(sys.argv[2]).resolve()
process_path = Path(process_data.__file__).resolve()
gr00t_path = Path(gr00t.__file__).resolve()
if not process_path.is_relative_to(xpl_root):
    raise RuntimeError(f"XPolicyLab resolved outside current repository: {process_path}")
if not gr00t_path.is_relative_to(gr00t_root):
    raise RuntimeError(f"gr00t resolved outside current repository: {gr00t_path}")
print(f"XPolicyLab source: {process_path}")
print(f"gr00t source: {gr00t_path}")
PY

dataset_ready=0
if [[ -d "${dataset_path}" ]]; then
  if [[ ! -f "${dataset_auditor}" ]]; then
    echo "Spark0 dataset auditor not found: ${dataset_auditor}" >&2
    exit 1
  fi
  "${GR00T_PYTHON}" - "${dataset_path}" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))
assert info["codebase_version"] == "v2.1", info["codebase_version"]
assert info["fps"] == 25, info["fps"]
assert info["total_episodes"] == 600, info["total_episodes"]
assert info["total_frames"] == 151410, info["total_frames"]
assert info["total_tasks"] == 6, info["total_tasks"]
assert info["features"]["observation.state"]["shape"] == [54]
assert info["features"]["action"]["shape"] == [54]
expected = ["left_arm", "left_hand", "right_arm", "right_hand"]
assert list(modality["state"]) == expected, list(modality["state"])
assert list(modality["action"]) == expected, list(modality["action"])
assert (dataset / "meta/stats.json").is_file()
assert (dataset / "meta/relative_stats.json").is_file()
print("Spark0 dataset preflight: v2.1, 25 Hz, 600 episodes, 151410 frames, joint54")
PY
  "${GR00T_PYTHON}" "${dataset_auditor}" \
    --dataset "${dataset_path}" \
    --config-path "${modality_config}" \
    --verify-marker-only \
    --require-full
  dataset_ready=1
fi

export NUM_GPUS=8
export GLOBAL_BATCH_SIZE=64
export MAX_STEPS=100000
export SAVE_STEPS=5000
export USE_WANDB=1
export WANDB_MODE=online
export GR00T_BASE_MODEL="${base_model}"
export GR00T_COSMOS_MODEL="${cosmos_model}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "[GR00T_N17] run=${ckpt_setting}"
echo "[GR00T_N17] dataset_path=${dataset_path}"
echo "[GR00T_N17] output_dir=${output_dir}"
echo "[GR00T_N17] base_model=${base_model}"
echo "[GR00T_N17] cosmos_model=${cosmos_model}"
echo "[GR00T_N17] gpu_id=${gpu_id} num_gpus=${NUM_GPUS}"
echo "[GR00T_N17] global_batch_size=${GLOBAL_BATCH_SIZE} per_gpu_batch_size=8"
echo "[GR00T_N17] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS}"
echo "[GR00T_N17] use_wandb=${USE_WANDB} wandb_mode=${WANDB_MODE}"
echo "[GR00T_N17] huggingface_offline=${HF_HUB_OFFLINE}"
echo "[GR00T_N17] dataset_ready=${dataset_ready}"

if [[ "${GR00T_TRAIN_PREVIEW:-0}" == "1" ]]; then
  echo "[GR00T_N17] preview only; training was not started."
  exit 0
fi
if (( dataset_ready != 1 )); then
  echo "Processed Spark0 joint54 dataset not found or incomplete: ${dataset_path}" >&2
  echo "Run process_data.sh first." >&2
  exit 1
fi
if [[ ! -f "${modality_config}" ]]; then
  echo "Modality config not found: ${modality_config}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the 8-GPU launch preflight." >&2
  exit 1
fi
for gpu in "${gpu_ids[@]}"; do
  compute_apps="$(
    nvidia-smi -i "${gpu}" \
      --query-compute-apps=pid,process_name \
      --format=csv,noheader,nounits
  )"
  if [[ -n "${compute_apps//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} already has compute applications; refusing to collide:" >&2
    printf '%s\n' "${compute_apps}" >&2
    exit 1
  fi
done

exec "${POLICY_DIR}/train.sh" \
  "${bench_name}" \
  "${ckpt_name}" \
  "${env_cfg_type}" \
  "${action_type}" \
  "${seed}" \
  "${gpu_id}"
