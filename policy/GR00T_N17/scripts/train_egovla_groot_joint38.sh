#!/usr/bin/env bash
# Fine-tune GR00T N1.7 on all active EgoVLA tasks (38-D manipulation space).
#
# Example:
#   WANDB_API_KEY=... bash train_egovla_groot_joint38.sh

set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
MODEL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
DATASET_PATH="${EGO_VLA_DATASET_PATH:-${MODEL_ROOT}/data/EgoVLA_benchmark_raw_action_v21}"
BASE_MODEL="${GR00T_BASE_MODEL:-${MODEL_ROOT}/pretrain_model/GR00T-N1.7-3B}"
COSMOS_MODEL="${GR00T_COSMOS_MODEL:-${MODEL_ROOT}/pretrain_model/Cosmos-Reason2-2B}"
MODALITY_CONFIG="${POLICY_DIR}/configs/ego_h1_inspire_config.py"
EGOVLA_INTEGRATION_SRC="${EGOVLA_INTEGRATION_SRC:-/personal/xiangpc/EgoVLA benchmark/integration/src}"
OBSERVATION_PROFILE_SCRIPT="${EGOVLA_INTEGRATION_SRC}/egovla_xpolicy/observation_profile.py"
RUN_NAME="${GR00T_RUN_NAME:-EgoVLA-all_tasks-ego_h1_inspire-joint-38d-raw-action-seed0-20260920}"
OUTPUT_ROOT="${GR00T_OUTPUT_ROOT:-${POLICY_DIR}/checkpoints}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
GPU_COUNT="$(tr ',' '\n' <<< "${CUDA_VISIBLE_DEVICES}" | sed '/^$/d' | wc -l | xargs)"
export NUM_GPUS="${NUM_GPUS:-${GPU_COUNT}}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
export MAX_STEPS="${MAX_STEPS:-80000}"
export SAVE_STEPS="${SAVE_STEPS:-10000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-8}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-finetune-gr00t-n1d7-egovla}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
export SHARD_SIZE="${SHARD_SIZE:-1024}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
export STATE_DROPOUT_PROB="${STATE_DROPOUT_PROB:-0.2}"
export GR00T_VIDEO_BACKEND="${GR00T_VIDEO_BACKEND:-pyav}"
# The converter records a lossless frame-count audit for all 5,709 videos;
# avoid repeating one ffprobe subprocess per camera/episode in every rank.
export GR00T_TRUST_VIDEO_LENGTHS="${GR00T_TRUST_VIDEO_LENGTHS:-1}"
export MASTER_PORT="${MASTER_PORT:-29517}"

# ffprobe is required by the GR00T episode synchronisation check even when the
# actual decoder is PyAV.  uv is not needed for this direct launcher.
export PATH="/personal/miniconda3/bin:/personal/miniconda3/envs/Luminis/bin:${PATH}"
export GR00T_COSMOS_MODEL="${COSMOS_MODEL}"
# Prefer the GR00T checkout that belongs to this XPolicyLab model tree over an
# editable package that may be installed in the shared base environment.
export PYTHONPATH="${GR00T_ROOT}:${MODEL_ROOT}:${PYTHONPATH:-}"

if [[ "${NUM_GPUS}" -ne "${GPU_COUNT}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS} must match the number of visible GPUs (${GPU_COUNT})." >&2
  exit 2
fi
if [[ "${GLOBAL_BATCH_SIZE}" -le 0 || $((GLOBAL_BATCH_SIZE % NUM_GPUS)) -ne 0 ]]; then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
if [[ $((GLOBAL_BATCH_SIZE / NUM_GPUS)) -ne 8 ]]; then
  echo "Expected per-card batch size 8, got $((GLOBAL_BATCH_SIZE / NUM_GPUS))." >&2
  exit 2
fi
for required in "${DATASET_PATH}/meta/info.json" "${DATASET_PATH}/meta/episodes.jsonl" \
  "${DATASET_PATH}/meta/tasks.jsonl" "${DATASET_PATH}/meta/modality.json" \
  "${DATASET_PATH}/meta/stats.json" "${DATASET_PATH}/meta/relative_stats.json" \
  "${DATASET_PATH}/meta/egovla_groot_conversion.json" \
  "${BASE_MODEL}/config.json" "${BASE_MODEL}/processor_config.json" \
  "${COSMOS_MODEL}/config.json" "${MODALITY_CONFIG}" "${OBSERVATION_PROFILE_SCRIPT}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 1; }
done
if [[ "${GRADIENT_ACCUMULATION_STEPS}" -ne 1 ]]; then
  echo "GRADIENT_ACCUMULATION_STEPS must be 1 so the optimizer batch remains exactly 64." >&2
  exit 2
fi
if [[ "${SAVE_TOTAL_LIMIT}" -lt 8 ]]; then
  echo "SAVE_TOTAL_LIMIT must be at least 8 to retain checkpoints 10000 through 80000." >&2
  exit 2
fi
if [[ "${USE_WANDB}" == "1" && -z "${WANDB_API_KEY:-}" ]] && \
  ! grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai' /root/.netrc 2>/dev/null; then
  echo "USE_WANDB=1 requires WANDB_API_KEY or an api.wandb.ai entry in /root/.netrc." >&2
  exit 1
fi

RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ -e "${RUN_OUTPUT}" ]]; then
  echo "Refusing to merge a fresh run into existing output: ${RUN_OUTPUT}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${RUN_OUTPUT}"
echo "[EgoVLA GR00T] dataset=${DATASET_PATH}"
echo "[EgoVLA GR00T] run=${RUN_NAME} output=${OUTPUT_ROOT}/${RUN_NAME}"
echo "[EgoVLA GR00T] GPUs=${CUDA_VISIBLE_DEVICES} global_bs=${GLOBAL_BATCH_SIZE} per_gpu_bs=$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
echo "[EgoVLA GR00T] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT} wandb=${USE_WANDB}"

PYTHONPATH="${EGOVLA_INTEGRATION_SRC}:${PYTHONPATH:-}" \
  "${GR00T_ROOT}/.venv/bin/python" -m egovla_xpolicy.observation_profile \
  --dataset "${DATASET_PATH}" \
  --output "${RUN_OUTPUT}/egovla_observation.json"

cd "${GR00T_ROOT}"
source .venv/bin/activate

python - "${DATASET_PATH}" "${RUN_OUTPUT}" "${BASE_MODEL}" "${COSMOS_MODEL}" \
  "${MODALITY_CONFIG}" "${NUM_GPUS}" "${GLOBAL_BATCH_SIZE}" "${MAX_STEPS}" \
  "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${USE_WANDB}" "${WANDB_PROJECT}" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
run_output = Path(sys.argv[2])
base_model = Path(sys.argv[3])
cosmos_model = Path(sys.argv[4])
modality_config = Path(sys.argv[5])
num_gpus, global_batch_size, max_steps = map(int, sys.argv[6:9])
save_steps, save_total_limit, grad_accum = map(int, sys.argv[9:12])
use_wandb = sys.argv[12] == "1"
wandb_project = sys.argv[13]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = json.loads((dataset / "meta/egovla_groot_conversion.json").read_text())
for name, expected_sha256 in manifest["dataset_meta_sha256"].items():
    metadata_path = dataset / "meta" / name
    assert sha256(metadata_path) == expected_sha256, (
        name,
        sha256(metadata_path),
        expected_sha256,
    )
audit = manifest["raw_action_rewrite"]
assert audit["training_action_source"] == "raw HDF5 /action at the same timestep"
assert audit["next_observed_state_used_as_action"] is False
assert audit["episodes_rewritten"] == 1903
assert audit["frames_rewritten"] == 510546
assert manifest["action_type"] == "joint"
assert manifest["action_dim"] == 38
assert manifest["action_horizon"] == 16
assert manifest["prompts"]["Insert-And-Unload-Cans"].endswith(
    "unload the left cans andd then unload the right cans"
)
assert manifest["action_representation"] == {
    "left_arm": "relative to same-timestep state (processor transform)",
    "right_arm": "relative to same-timestep state (processor transform)",
    "left_hand": "absolute",
    "right_hand": "absolute",
}
modality = json.loads((dataset / "meta/modality.json").read_text())
for name, spec in modality["action"].items():
    assert spec.get("original_key", "action") == "action", (name, spec)

profile_path = run_output / "egovla_observation.json"
profile = json.loads(profile_path.read_text())
assert profile["schema"] == "egovla-observation-profile-v1"
assert profile["color_order"] == "RGB"
assert profile["camera_shapes"] == {
    "cam_head": [384, 384],
    "cam_left_wrist": [384, 384],
    "cam_right_wrist": [384, 384],
}
assert Path(profile["source"]["dataset"]).resolve() == dataset.resolve()
assert len(profile["task_camera_masks"]) == 12
real_wrist_tasks = sorted(
    task for task, camera_mask in profile["task_camera_masks"].items()
    if camera_mask == [True, True, True]
)
assert real_wrist_tasks == ["Humanoid-Insert-And-Unload-Cans-v0"]
assert all(
    camera_mask in ([True, False, False], [True, True, True])
    for camera_mask in profile["task_camera_masks"].values()
)

processor_path = base_model / "processor_config.json"
processor_kwargs = json.loads(processor_path.read_text())["processor_kwargs"]
processor_contract = {
    "image_target_size": processor_kwargs["image_target_size"],
    "image_crop_size": processor_kwargs["image_crop_size"],
    "shortest_image_edge": processor_kwargs["shortest_image_edge"],
    "crop_fraction": processor_kwargs["crop_fraction"],
    "formalize_language": processor_kwargs["formalize_language"],
}
assert processor_contract == {
    "image_target_size": [256, 256],
    "image_crop_size": [230, 230],
    "shortest_image_edge": 256,
    "crop_fraction": 0.95,
    "formalize_language": True,
}

contract = {
    "schema": "egovla-groot-training-contract-v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "dataset": {
        "path": str(dataset.resolve()),
        "conversion_manifest_sha256": sha256(dataset / "meta/egovla_groot_conversion.json"),
        "raw_dataset_manifest_sha256": manifest["raw_audit"]["dataset_manifest_sha256"],
        "raw_inventory_sha256": manifest["raw_audit"]["inventory_sha256"],
        "raw_action38_stream_sha256": audit["raw_action38_stream_sha256"],
        "episodes": audit["episodes_rewritten"],
        "frames": audit["frames_rewritten"],
    },
    "action": {
        "source": audit["training_action_source"],
        "next_observed_state_used_as_action": False,
        "type": manifest["action_type"],
        "dimension": manifest["action_dim"],
        "horizon": manifest["action_horizon"],
        "representation": manifest["action_representation"],
    },
    "observation": {
        "profile_path": str(profile_path.resolve()),
        "profile_sha256": sha256(profile_path),
        "raw_camera_contract": manifest["camera_contract"],
        "processor": processor_contract,
    },
    "prompts": manifest["prompts"],
    "pretrained": {
        "groot_n17_3b": {
            "path": str(base_model.resolve()),
            "config_sha256": sha256(base_model / "config.json"),
            "processor_config_sha256": sha256(processor_path),
        },
        "cosmos_reason2_2b": {
            "path": str(cosmos_model.resolve()),
            "config_sha256": sha256(cosmos_model / "config.json"),
        },
    },
    "training": {
        "modality_config": str(modality_config.resolve()),
        "modality_config_sha256": sha256(modality_config),
        "num_gpus": num_gpus,
        "global_batch_size": global_batch_size,
        "per_gpu_batch_size": global_batch_size // num_gpus,
        "gradient_accumulation_steps": grad_accum,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "wandb_enabled": use_wandb,
        "wandb_project": wandb_project,
    },
}
contract_path = run_output / "egovla_training_contract.json"
with contract_path.open("x", encoding="utf-8") as stream:
    json.dump(contract, stream, ensure_ascii=False, indent=2, sort_keys=True)
    stream.write("\n")
print("[EgoVLA GR00T] raw-action, camera, prompt, processor, and pretrained contracts verified")
PY

WANDB_FLAG=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_FLAG+=(--use-wandb)
fi

exec torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
  gr00t/experiment/launch_finetune.py \
  --base-model-path "${BASE_MODEL}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path "${MODALITY_CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --output-dir "${OUTPUT_ROOT}" \
  --experiment-name "${RUN_NAME}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --max-steps "${MAX_STEPS}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --warmup-ratio 0.05 \
  --weight-decay 1e-5 \
  --learning-rate 1e-4 \
  --state-dropout-prob "${STATE_DROPOUT_PROB}" \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --shard-size "${SHARD_SIZE}" \
  --num-shards-per-epoch "${NUM_SHARDS_PER_EPOCH}" \
  --episode-sampling-rate "${EPISODE_SAMPLING_RATE}" \
  --wandb-project "${WANDB_PROJECT}" \
  "${WANDB_FLAG[@]}"
