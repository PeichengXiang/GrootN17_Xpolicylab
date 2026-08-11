#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [episodes_per_task]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
episodes_per_task=${5:-100}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
GR00T_ROOT="${POLICY_DIR}/gr00t_n17"
GR00T_PYTHON="${GR00T_ROOT}/.venv/bin/python"
CONVERTER_PYTHON="${GR00T_LEROBOT_V21_PYTHON:-/personal/miniconda3/envs/dexora_1b/bin/python}"
SOURCE_ROOT="${GR00T_SPARK0_HDF5_ROOT:-/mnt/xspark-data/tjy/spark0_bench}"
DATA_ROOT="${GR00T_LEROBOT_HOME:-}"
MODALITY_JSON="${POLICY_DIR}/configs/tianji_marvin_wuji_modality.json"
MODALITY_CONFIG="${POLICY_DIR}/configs/tianji_marvin_wuji_config.py"
CHECKER="${XPL_ROOT}/scripts/check_groot_joint54_compatibility.py"
CONVERTER="${POLICY_DIR}/scripts/convert_spark0_hdf5_to_groot_v21.py"
AUDITOR="${POLICY_DIR}/scripts/audit_spark0_groot_v21.py"
TASKS=(collect_objects dual_bottles_pick hammer_beat insert_block retrieve_gap stack_bowls)

if [[ -z "${DATA_ROOT}" ]]; then
  echo "Set GR00T_LEROBOT_HOME to the LeRobot datasets root." >&2
  exit 1
fi
if [[ "${bench_name}" != "Spark0_bench" ]]; then
  echo "Spark0 joint54 conversion requires bench_name=Spark0_bench, got '${bench_name}'." >&2
  exit 1
fi
if [[ "${env_cfg_type}" != "tianji_marvin_wuji" || "${action_type}" != "joint" ]]; then
  echo "Spark0 joint54 conversion requires env_cfg_type=tianji_marvin_wuji and action_type=joint." >&2
  exit 1
fi
for identifier in "${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}"; do
  if [[ ! "${identifier}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Unsafe run identifier: '${identifier}'. Use only letters, digits, '_' and '-'." >&2
    exit 1
  fi
done
if [[ ! "${episodes_per_task}" =~ ^[1-9][0-9]*$ ]] || (( episodes_per_task > 100 )); then
  echo "episodes_per_task must be an integer from 1 to 100, got '${episodes_per_task}'." >&2
  exit 1
fi

SOURCE_ROOT="$(realpath -e "${SOURCE_ROOT}")"
case "${SOURCE_ROOT}" in
  */sim_6tasks_lerobot|*/sim_6tasks_lerobot/*)
    echo "Refusing converted input ${SOURCE_ROOT}; pass the raw Spark0 HDF5 root." >&2
    exit 1
    ;;
esac

for required_path in \
  "${GR00T_PYTHON}" \
  "${CONVERTER_PYTHON}" \
  "${MODALITY_JSON}" \
  "${MODALITY_CONFIG}" \
  "${CHECKER}" \
  "${CONVERTER}" \
  "${AUDITOR}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path not found: ${required_path}" >&2
    exit 1
  fi
done
for task in "${TASKS[@]}"; do
  if [[ ! -d "${SOURCE_ROOT}/${task}/tianji_marvin_wuji/data" ]]; then
    echo "Missing raw task data for ${task} under ${SOURCE_ROOT}." >&2
    exit 1
  fi
done

mkdir -p "${DATA_ROOT}"
DATA_ROOT="$(realpath -e "${DATA_ROOT}")"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
dataset_path="${DATA_ROOT}/${data_setting}"
if [[ -e "${dataset_path}" || -L "${dataset_path}" ]]; then
  echo "Output dataset already exists; refusing reconversion: ${dataset_path}" >&2
  exit 1
fi

export PYTHONPATH="${XPL_ROOT}:${GR00T_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

assert_current_imports() {
  local python_bin=$1
  "${python_bin}" - "${XPL_ROOT}" "${GR00T_ROOT}" <<'PY'
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
}

echo "[GR00T_N17] source_root=${SOURCE_ROOT}"
echo "[GR00T_N17] output_dataset=${dataset_path}"
echo "[GR00T_N17] physical_schema=left_arm:7,left_hand:20,right_arm:7,right_hand:20"
echo "[GR00T_N17] processor_order=left_arm,right_arm,left_hand,right_hand"
echo "[GR00T_N17] tasks=${TASKS[*]} episodes_per_task=${episodes_per_task}"
echo "[GR00T_N17] fps=25 resolution=480x640 cameras=cam_high,cam_left_wrist,cam_right_wrist"

assert_current_imports "${GR00T_PYTHON}"
"${GR00T_PYTHON}" "${CHECKER}" --source-root "${SOURCE_ROOT}"

if [[ "${GR00T_PROCESS_PREVIEW:-0}" == "1" ]]; then
  echo "[GR00T_N17] preview only; no dataset was created."
  exit 0
fi

assert_current_imports "${CONVERTER_PYTHON}"
"${CONVERTER_PYTHON}" - <<'PY'
import importlib.metadata

version = importlib.metadata.version("lerobot")
if version != "0.3.3":
    raise RuntimeError(f"Expected lerobot==0.3.3 for v2.1 output, got {version}")
print(f"LeRobot converter version: {version}")
PY

lock_dir="${dataset_path}.conversion.lock"
lock_acquired=0
stage_root=""

cleanup() {
  if [[ -n "${stage_root:-}" && -e "${stage_root}" ]]; then
    resolved_stage_root="$(realpath -e "${stage_root}")"
    case "${resolved_stage_root}" in
      "${DATA_ROOT}"/.groot-spark0-*) rm -rf -- "${resolved_stage_root}" ;;
      *) echo "Refusing unsafe staging cleanup: ${resolved_stage_root}" >&2 ;;
    esac
  fi
  if (( lock_acquired == 1 )) && [[ -n "${lock_dir:-}" && -d "${lock_dir}" ]]; then
    rmdir -- "${lock_dir}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
if ! mkdir "${lock_dir}"; then
  echo "Another conversion owns the output lock: ${lock_dir}" >&2
  exit 1
fi
lock_acquired=1
stage_root="$(mktemp -d "${DATA_ROOT}/.groot-spark0-${data_setting}.XXXXXX")"
stage_dataset_root="${stage_root}/lerobot"
stage_dataset="${stage_dataset_root}/${data_setting}"
mkdir "${stage_dataset_root}"

HF_LEROBOT_HOME="${stage_dataset_root}" "${CONVERTER_PYTHON}" "${CONVERTER}" \
  --source-root "${SOURCE_ROOT}" \
  --output "${stage_dataset}" \
  --repo-root "${XPL_ROOT}" \
  --episodes-per-task "${episodes_per_task}" \
  --image-writer-threads "${GR00T_IMAGE_WRITER_THREADS:-8}"

if [[ ! -d "${stage_dataset}/meta" ]]; then
  echo "Converter did not create expected staging dataset: ${stage_dataset}" >&2
  exit 1
fi
install -m 0644 "${MODALITY_JSON}" "${stage_dataset}/meta/modality.json"

"${GR00T_PYTHON}" - "${stage_dataset}" "$((episodes_per_task * 6))" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
expected_episodes = int(sys.argv[2])
info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))
manifest = json.loads(
    (dataset / "meta/spark0_source_manifest.json").read_text(encoding="utf-8")
)
features = info["features"]
assert info["codebase_version"] == "v2.1", info["codebase_version"]
assert info["fps"] == 25, info["fps"]
assert info["total_episodes"] == expected_episodes, info["total_episodes"]
assert info["total_tasks"] == 6, info["total_tasks"]
if expected_episodes == 600:
    assert info["total_frames"] == 151410, info["total_frames"]
assert len(manifest) == expected_episodes, len(manifest)
assert features["observation.state"]["shape"] == [54]
assert features["action"]["shape"] == [54]
expected_videos = {
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
}
actual_videos = {key for key, value in features.items() if value["dtype"] == "video"}
assert actual_videos == expected_videos, actual_videos
expected_ranges = {
    "left_arm": (0, 7),
    "left_hand": (7, 27),
    "right_arm": (27, 34),
    "right_hand": (34, 54),
}
for section in ("state", "action"):
    actual = {
        key: (value["start"], value["end"])
        for key, value in modality[section].items()
    }
    assert actual == expected_ranges, (section, actual)
print(
    "Validated staging dataset: "
    f"version=v2.1 fps=25 episodes={expected_episodes} tasks=6 dim=54 videos=3"
)
PY

(
  cd "${GR00T_ROOT}"
  "${GR00T_PYTHON}" gr00t/data/stats.py \
    --dataset-path "${stage_dataset}" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIG}"
)

if [[ ! -f "${stage_dataset}/meta/stats.json" \
   || ! -f "${stage_dataset}/meta/relative_stats.json" ]]; then
  echo "GR00T absolute/relative statistics were not generated under ${stage_dataset}/meta." >&2
  exit 1
fi

audit_args=(
  --dataset "${stage_dataset}"
  --source-root "${SOURCE_ROOT}"
  --config-path "${MODALITY_CONFIG}"
  --repo-root "${XPL_ROOT}"
  --expected-episodes "$((episodes_per_task * 6))"
  --ffprobe-workers "${GR00T_AUDIT_FFPROBE_WORKERS:-8}"
)
if (( episodes_per_task == 100 )); then
  audit_args+=(--require-full)
fi
"${GR00T_PYTHON}" "${AUDITOR}" "${audit_args[@]}"

if [[ -e "${dataset_path}" || -L "${dataset_path}" ]]; then
  echo "Output appeared during conversion; refusing publish: ${dataset_path}" >&2
  exit 1
fi
mv -T -- "${stage_dataset}" "${dataset_path}"
echo "[GR00T_N17] Spark0 joint54 conversion complete: ${dataset_path}"
