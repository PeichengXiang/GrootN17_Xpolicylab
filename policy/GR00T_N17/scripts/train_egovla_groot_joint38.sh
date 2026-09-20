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
OBSERVATION_PROFILE_SCRIPT="${EGOVLA_OBSERVATION_PROFILE_SCRIPT:-${MODEL_ROOT}/data_scripts/profile_egovla_observations.py}"
RUN_NAME="${GR00T_RUN_NAME:-EgoVLA-all_tasks-ego_h1_inspire-joint-38d-raw-action-seed42-20260920}"
OUTPUT_ROOT="${GR00T_OUTPUT_ROOT:-${POLICY_DIR}/checkpoints}"
TRAIN_SEED=42
RESUME_MODE="${GR00T_RESUME:-0}"

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
export PYTHONOPTIMIZE=0

# ffprobe is required by the GR00T episode synchronisation check even when the
# actual decoder is PyAV.  Keep host-specific PATH additions opt-in.
if [[ -n "${GR00T_EXTRA_PATH:-}" ]]; then
  export PATH="${GR00T_EXTRA_PATH}:${PATH}"
fi
command -v ffprobe >/dev/null 2>&1 || {
  echo "ffprobe is required but was not found on PATH." >&2
  exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
  echo "nvidia-smi is required for the 8-GPU launch preflight." >&2
  exit 1
}
require_idle_gpus() {
  local active_pids
  active_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${active_pids}" ]]; then
    echo "Refusing to launch while GPU compute processes are active: ${active_pids//$'\n'/,}" >&2
    return 1
  fi
}
require_idle_gpus
export GR00T_COSMOS_MODEL="${COSMOS_MODEL}"
# Prefer the GR00T checkout that belongs to this XPolicyLab model tree over an
# editable package that may be installed in the shared base environment.
export PYTHONPATH="${GR00T_ROOT}:${MODEL_ROOT}:${PYTHONPATH:-}"

if [[ "${NUM_GPUS}" -ne "${GPU_COUNT}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS} must match the number of visible GPUs (${GPU_COUNT})." >&2
  exit 2
fi
if [[ "${NUM_GPUS}" -ne 8 ]]; then
  echo "This run requires exactly 8 GPUs, got NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
if [[ "${GLOBAL_BATCH_SIZE}" -le 0 || $((GLOBAL_BATCH_SIZE % NUM_GPUS)) -ne 0 ]]; then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}." >&2
  exit 2
fi
if [[ "${GLOBAL_BATCH_SIZE}" -ne 64 ]]; then
  echo "This run requires GLOBAL_BATCH_SIZE=64, got ${GLOBAL_BATCH_SIZE}." >&2
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
if [[ "${MAX_STEPS}" -ne 80000 || "${SAVE_STEPS}" -ne 10000 ]]; then
  echo "This run requires MAX_STEPS=80000 and SAVE_STEPS=10000." >&2
  exit 2
fi
if [[ "${SAVE_TOTAL_LIMIT}" -ne 8 ]]; then
  echo "SAVE_TOTAL_LIMIT must be 8 to retain checkpoints 10000 through 80000." >&2
  exit 2
fi
if [[ "${USE_WANDB}" != "1" || "${WANDB_MODE}" != "online" ]]; then
  echo "This run requires USE_WANDB=1 and WANDB_MODE=online." >&2
  exit 2
fi
if [[ "${RESUME_MODE}" != "0" && "${RESUME_MODE}" != "1" ]]; then
  echo "GR00T_RESUME must be 0 or 1, got ${RESUME_MODE}." >&2
  exit 2
fi
if [[ -z "${WANDB_API_KEY:-}" ]] && \
  ! grep -Eq '^[[:space:]]*machine[[:space:]]+api\.wandb\.ai' /root/.netrc 2>/dev/null; then
  echo "W&B online mode requires WANDB_API_KEY or an api.wandb.ai entry in /root/.netrc." >&2
  exit 1
fi

RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_NAME}"
NEW_RUN=1
PROFILE_CHECK_PATH=""
if [[ -e "${RUN_OUTPUT}" ]]; then
  if [[ "${RESUME_MODE}" != "1" ]]; then
    echo "Existing output requires explicit GR00T_RESUME=1: ${RUN_OUTPUT}" >&2
    exit 1
  fi
  for required in "${RUN_OUTPUT}/egovla_observation.json" \
    "${RUN_OUTPUT}/egovla_training_contract.json"; do
    [[ -f "${required}" ]] || { echo "Resume sidecar is missing: ${required}" >&2; exit 1; }
  done
  RESUME_CHECKPOINT="$("${GR00T_ROOT}/.venv/bin/python" - "${RUN_OUTPUT}" <<'PY'
from pathlib import Path
import re
import sys

run_output = Path(sys.argv[1])
candidates = []
for path in run_output.iterdir():
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if path.is_dir() and match:
        candidates.append((int(match.group(1)), path))
if not candidates:
    raise FileNotFoundError(f"no numeric checkpoint-* directory under {run_output}")
checkpoint = max(candidates)[1]
for name in (
    "trainer_state.json",
    "config.json",
    "processor_config.json",
    "egovla_training_contract.json",
    "egovla_observation.json",
):
    if not (checkpoint / name).is_file():
        raise FileNotFoundError(f"latest checkpoint is incomplete: {checkpoint / name}")
if not (
    (checkpoint / "model.safetensors").is_file()
    or (checkpoint / "model.safetensors.index.json").is_file()
):
    raise FileNotFoundError(f"latest checkpoint has no model weights: {checkpoint}")
deepspeed_files = [path for path in checkpoint.glob("global_step*/**/*") if path.is_file()]
if not any(path.name.endswith("_optim_states.pt") for path in deepspeed_files):
    raise FileNotFoundError(f"latest checkpoint has no DeepSpeed optimizer state: {checkpoint}")
if not any(path.name.endswith("_model_states.pt") for path in deepspeed_files):
    raise FileNotFoundError(f"latest checkpoint has no DeepSpeed model/scheduler state: {checkpoint}")
print(checkpoint)
PY
)"
  NEW_RUN=0
  PROFILE_CHECK_PATH="${RUN_OUTPUT}/.egovla_observation.resume-check-$$-${RANDOM}.json"
  if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_ID="$("${GR00T_ROOT}/.venv/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["training"]["wandb_run_id"])' \
      "${RUN_OUTPUT}/egovla_training_contract.json")"
  fi
else
  if [[ "${RESUME_MODE}" == "1" ]]; then
    echo "GR00T_RESUME=1 was requested but the run output does not exist: ${RUN_OUTPUT}" >&2
    exit 1
  fi
  WANDB_RUN_ID="${WANDB_RUN_ID:-$("${GR00T_ROOT}/.venv/bin/python" -c 'import secrets; print(secrets.token_hex(8))')}"
fi
export WANDB_RUN_ID
export WANDB_RESUME=allow
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${RUN_OUTPUT}"
TRAIN_LAUNCHED=0
retain_failed_preflight() {
  if [[ -n "${PROFILE_CHECK_PATH}" ]]; then
    rm -f "${PROFILE_CHECK_PATH}" || true
  fi
  if [[ "${TRAIN_LAUNCHED}" == "0" && "${NEW_RUN}" == "1" && -d "${RUN_OUTPUT}" ]]; then
    failed_output="${RUN_OUTPUT}.preflight-failed-$(date -u +%Y%m%dT%H%M%SZ)"
    mv "${RUN_OUTPUT}" "${failed_output}" || true
    echo "Preflight failed; diagnostics retained at ${failed_output}" >&2
  fi
}
trap retain_failed_preflight ERR
echo "[EgoVLA GR00T] dataset=${DATASET_PATH}"
echo "[EgoVLA GR00T] run=${RUN_NAME} output=${OUTPUT_ROOT}/${RUN_NAME}"
echo "[EgoVLA GR00T] resume=${RESUME_MODE} wandb_run_id=${WANDB_RUN_ID}"
if [[ "${RESUME_MODE}" == "1" ]]; then
  echo "[EgoVLA GR00T] latest complete checkpoint=${RESUME_CHECKPOINT}"
fi
echo "[EgoVLA GR00T] GPUs=${CUDA_VISIBLE_DEVICES} global_bs=${GLOBAL_BATCH_SIZE} per_gpu_bs=$((GLOBAL_BATCH_SIZE / NUM_GPUS))"
echo "[EgoVLA GR00T] max_steps=${MAX_STEPS} save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT} wandb=${USE_WANDB}"

if [[ "${RESUME_MODE}" == "1" ]]; then
  "${GR00T_ROOT}/.venv/bin/python" "${OBSERVATION_PROFILE_SCRIPT}" \
    --dataset "${DATASET_PATH}" \
    --output "${PROFILE_CHECK_PATH}"
else
  "${GR00T_ROOT}/.venv/bin/python" "${OBSERVATION_PROFILE_SCRIPT}" \
    --dataset "${DATASET_PATH}" \
    --output "${RUN_OUTPUT}/egovla_observation.json"
fi

cd "${GR00T_ROOT}"
source .venv/bin/activate

python - "${DATASET_PATH}" "${RUN_OUTPUT}" "${BASE_MODEL}" "${COSMOS_MODEL}" \
  "${MODALITY_CONFIG}" "${NUM_GPUS}" "${GLOBAL_BATCH_SIZE}" "${MAX_STEPS}" \
  "${SAVE_STEPS}" "${SAVE_TOTAL_LIMIT}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${USE_WANDB}" "${WANDB_PROJECT}" "${TRAIN_SEED}" "${WANDB_MODE}" \
  "${WANDB_RUN_ID}" "${RESUME_MODE}" "${PROFILE_CHECK_PATH}" <<'PY'
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from gr00t.configs.data.data_config import DataConfig

dataset = Path(sys.argv[1])
run_output = Path(sys.argv[2])
base_model = Path(sys.argv[3])
cosmos_model = Path(sys.argv[4])
modality_config = Path(sys.argv[5])
num_gpus, global_batch_size, max_steps = map(int, sys.argv[6:9])
save_steps, save_total_limit, grad_accum = map(int, sys.argv[9:12])
use_wandb = sys.argv[12] == "1"
wandb_project = sys.argv[13]
training_seed = int(sys.argv[14])
wandb_mode = sys.argv[15]
wandb_run_id = sys.argv[16]
resume_mode = sys.argv[17] == "1"
profile_check_path = Path(sys.argv[18]) if sys.argv[18] else None

if sys.flags.optimize != 0:
    raise RuntimeError("EgoVLA contract checks require Python optimization to be disabled")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

manifest = json.loads((dataset / "meta/egovla_groot_conversion.json").read_text())
assert training_seed == 42
assert DataConfig().seed == training_seed
assert use_wandb and wandb_mode == "online"
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
assert audit["raw_action38_stream_sha256"] == (
    "96dfbce561c21dd413845c50b05da6ce77f1a2b0e8dd2312bc25f85de4ebdfd2"
)
assert audit["raw_state38_stream_sha256"] == (
    "78f21e4f2376c85d8deed8795ea190c87ee29b941e84834e8efe7814b4e80a7e"
)
assert manifest["action_type"] == "joint"
assert manifest["action_dim"] == 38
assert manifest["action_horizon"] == 16
expected_prompts = {
    "Pour-Balls": "pour balls in cup into bowl",
    "Push-Box": "push box to the marker",
    "Sort-Cans": "Put sprite cans to the left box, and orange cans to the right box",
    "Insert-Cans": "Insert cans into the boxes",
    "Close-Drawer": "Close the opened drawer",
    "Open-Drawer": "Open the closed drawer",
    "Insert-And-Unload-Cans": (
        "Insert the left can into the slot and insert the right can into the slot, "
        "unload the left cans andd then unload the right cans"
    ),
    "Flip-Mug": "Flip the mug",
    "Unload-Cans": "unload the right cans and then unload the left cans",
    "Stack-Can": "put can on the saucer",
    "Stack-Can-Into-Drawer": "Open the drawer, and Put can on the saucer",
    "Open-Laptop": "open the laptop",
}
assert manifest["prompts"] == expected_prompts
tasks = [
    json.loads(line)
    for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()
    if line.strip()
]
assert [int(row["task_index"]) for row in tasks] == list(range(12))
assert len({row["task"] for row in tasks}) == 12
assert {row["task"] for row in tasks} == set(expected_prompts.values())
assert manifest["action_representation"] == {
    "left_arm": "relative to same-timestep state (processor transform)",
    "right_arm": "relative to same-timestep state (processor transform)",
    "left_hand": "absolute",
    "right_hand": "absolute",
}
assert sha256(modality_config) == (
    "fcdadfac0d6a94aacdd33ae07f5b87ef18e01b55926c50f3a42980ff454b2f86"
)
assert manifest["modality_config_sha256"] == sha256(modality_config)
modality = json.loads((dataset / "meta/modality.json").read_text())
for name, spec in modality["action"].items():
    assert spec.get("original_key", "action") == "action", (name, spec)

profile_path = run_output / "egovla_observation.json"
profile = json.loads(profile_path.read_text())
if resume_mode:
    if profile_check_path is None or not profile_check_path.is_file():
        raise FileNotFoundError("resume observation re-audit was not produced")
    fresh_profile = json.loads(profile_check_path.read_text())
    stable_profile = dict(profile)
    stable_fresh_profile = dict(fresh_profile)
    stable_profile.pop("created_at_utc", None)
    stable_fresh_profile.pop("created_at_utc", None)
    if stable_profile != stable_fresh_profile:
        raise ValueError("resume dataset observation profile differs from the original run")
    profile_check_path.unlink()
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
assert sha256(base_model / "config.json") == (
    "54c0367060cd310d0b3343fe72a589860a8b6e8173810164a4ffd6253f52e689"
)
assert sha256(processor_path) == (
    "85c1b4690ae090559e79a45193e598b65d6146eedf14750884da65e6d31032be"
)
assert sha256(cosmos_model / "config.json") == (
    "bec4b3d446efa05807365c9e1cec03ac590836879d02f3a6da879971154bdd3b"
)
expected_weight_sha256 = {
    base_model / "model.safetensors.index.json": (
        "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59"
    ),
    base_model / "model-00001-of-00002.safetensors": (
        "8a1a1d8a33c99103c7c80c136073c5bb8bfe9ca8f7a970c93c033ea89742906d"
    ),
    base_model / "model-00002-of-00002.safetensors": (
        "c3f61940deb2007ba1ad7743013b57f0f8462356151db9655175d7aca2d40661"
    ),
    cosmos_model / "model.safetensors": (
        "fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede"
    ),
}
for weight_path, expected_sha256 in expected_weight_sha256.items():
    assert weight_path.is_file(), weight_path
    assert sha256(weight_path) == expected_sha256, weight_path
processor_kwargs = json.loads(processor_path.read_text())["processor_kwargs"]
processor_contract = {
    "image_target_size": processor_kwargs["image_target_size"],
    "image_crop_size": processor_kwargs["image_crop_size"],
    "shortest_image_edge": processor_kwargs["shortest_image_edge"],
    "crop_fraction": processor_kwargs["crop_fraction"],
    "formalize_language": processor_kwargs["formalize_language"],
    "use_relative_action": processor_kwargs["use_relative_action"],
    "apply_sincos_state_encoding": processor_kwargs["apply_sincos_state_encoding"],
    "exclude_state": processor_kwargs["exclude_state"],
    "use_mean_std": processor_kwargs["use_mean_std"],
    "use_percentiles": processor_kwargs["use_percentiles"],
}
assert processor_contract == {
    "image_target_size": [256, 256],
    "image_crop_size": [230, 230],
    "shortest_image_edge": 256,
    "crop_fraction": 0.95,
    "formalize_language": True,
    "use_relative_action": True,
    "apply_sincos_state_encoding": False,
    "exclude_state": False,
    "use_mean_std": False,
    "use_percentiles": True,
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
            "weight_files_sha256": {
                path.name: expected
                for path, expected in expected_weight_sha256.items()
                if path.parent == base_model
            },
        },
        "cosmos_reason2_2b": {
            "path": str(cosmos_model.resolve()),
            "config_sha256": sha256(cosmos_model / "config.json"),
            "weight_files_sha256": {
                path.name: expected
                for path, expected in expected_weight_sha256.items()
                if path.parent == cosmos_model
            },
        },
    },
    "training": {
        "modality_config": str(modality_config.resolve()),
        "modality_config_sha256": sha256(modality_config),
        "embodiment_tag": "NEW_EMBODIMENT",
        "num_gpus": num_gpus,
        "global_batch_size": global_batch_size,
        "per_gpu_batch_size": global_batch_size // num_gpus,
        "gradient_accumulation_steps": grad_accum,
        "seed": training_seed,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "wandb_enabled": use_wandb,
        "wandb_mode": wandb_mode,
        "wandb_project": wandb_project,
        "wandb_run_id": wandb_run_id,
    },
}
contract_path = run_output / "egovla_training_contract.json"
if resume_mode:
    existing_contract = json.loads(contract_path.read_text())
    contract["created_at_utc"] = existing_contract.get("created_at_utc")
    if contract != existing_contract:
        raise ValueError("resume training contract differs from the original run")
    print("[EgoVLA GR00T] resume contract and full dataset re-audit verified")
else:
    with contract_path.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
print("[EgoVLA GR00T] raw-action, camera, prompt, processor, and pretrained contracts verified")
PY

WANDB_FLAG=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_FLAG+=(--use-wandb)
fi

require_idle_gpus
TRAIN_LAUNCHED=1
trap - ERR
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
