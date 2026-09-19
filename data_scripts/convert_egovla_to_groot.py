#!/usr/bin/env python3
"""Prepare the raw-action EgoVLA benchmark for GR00T N1.7.

The source is the 50-DoF ``raw_remove_deprecated`` HDF5 release.  Existing
LeRobot v2.1 videos are reused byte-for-byte, but every parquet ``action``
column is rewritten and verified from the same-timestep raw HDF5 ``/action``
dataset using the canonical 50 -> 38 index map.  ``observations/qpos[t+1]`` is
never an action label.

The converter also proves that the template's provenance column is identical
to the source action, that state remains the same-timestep raw qpos, preserves
real wrist cameras only where present (black wrist videos elsewhere), and
regenerates absolute and relative statistics before publishing the output
atomically.  The resulting dataset contains 1,903 episodes, 510,546 frames,
12 canonical task prompts, RGB 384x384 videos at 30 Hz, and the 38-D
left-arm/left-hand/right-arm/right-hand layout used by XPolicyLab.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

try:
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:  # pragma: no cover - environment hint
    raise SystemExit(
        "h5py, numpy and pyarrow are required; run with policy/GR00T_N17/gr00t_n17/.venv/bin/python"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
DEFAULT_RAW = Path("/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw_remove_deprecated")
DEFAULT_CACHE = Path(
    "/personal/xiangpc/0811_Xpolicylab_bench/RLDX_1/data/EgoVLA_benchmark_rldx_v21"
)
DEFAULT_OUTPUT = MODEL_ROOT / "data" / "EgoVLA_benchmark_raw_action_v21"
DEFAULT_GR00T_ROOT = MODEL_ROOT / "policy" / "GR00T_N17" / "gr00t_n17"
DEFAULT_MODALITY_CONFIG = MODEL_ROOT / "policy" / "GR00T_N17" / "configs" / "ego_h1_inspire_config.py"

FPS = 30
IMAGE_SHAPE = (384, 384, 3)
RAW_DIM = 50
POLICY_DIM = 38
EXPECTED_DEPRECATED = 0
RAW_MANIFEST_SHA256 = "f3bfceb6fafd6f28b3d93d3f5048f20aae62871b34eff372b8e320aeda9d172c"
RAW_INVENTORY_SHA256 = "dc8265b266a9ebb0d3ad8264c1c7a810207cf5aa04b2ddca011d8470c1fa14df"
RAW_TOTAL_BYTES = 505_962_797_780

TASK_INSTRUCTIONS: dict[str, str] = {
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

# The reusable video template predates the benchmark-registry prompt audit and
# corrected the registry's historical "andd" typo.  It is accepted only as an
# input template; the published dataset is rewritten to the exact registry text.
TEMPLATE_TASK_INSTRUCTIONS = dict(TASK_INSTRUCTIONS)
TEMPLATE_TASK_INSTRUCTIONS["Insert-And-Unload-Cans"] = (
    "Insert the left can into the slot and insert the right can into the slot, "
    "unload the left cans and then unload the right cans"
)

# Counts are part of the official filtered-release contract.
EXPECTED_TASK_EPISODES = {
    "Close-Drawer": 50,
    "Flip-Mug": 100,
    "Insert-And-Unload-Cans": 900,
    "Insert-Cans": 100,
    "Open-Drawer": 100,
    "Open-Laptop": 100,
    "Pour-Balls": 102,
    "Push-Box": 100,
    "Sort-Cans": 101,
    "Stack-Can": 100,
    "Stack-Can-Into-Drawer": 50,
    "Unload-Cans": 100,
}
EXPECTED_TASK_FRAMES = {
    "Close-Drawer": 4120,
    "Flip-Mug": 9393,
    "Insert-And-Unload-Cans": 315999,
    "Insert-Cans": 28001,
    "Open-Drawer": 11168,
    "Open-Laptop": 10569,
    "Pour-Balls": 17129,
    "Push-Box": 15962,
    "Sort-Cans": 52670,
    "Stack-Can": 15096,
    "Stack-Can-Into-Drawer": 7020,
    "Unload-Cans": 23419,
}

RAW_EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$", re.IGNORECASE)

POLICY_JOINT_INDICES = (
    4,
    8,
    12,
    16,
    20,
    22,
    24,
    26,
    36,
    27,
    37,
    28,
    38,
    29,
    39,
    30,
    40,
    46,
    48,
    5,
    9,
    13,
    17,
    21,
    23,
    25,
    31,
    41,
    32,
    42,
    33,
    43,
    34,
    44,
    35,
    45,
    47,
    49,
)

GR00T_MODALITY: dict[str, Any] = {
    "state": {
        "left_arm": {"start": 0, "end": 7},
        "left_hand": {"start": 7, "end": 19},
        "right_arm": {"start": 19, "end": 26},
        "right_hand": {"start": 26, "end": 38},
    },
    "action": {
        "left_arm": {"start": 0, "end": 7},
        "left_hand": {"start": 7, "end": 19},
        "right_arm": {"start": 19, "end": 26},
        "right_hand": {"start": 26, "end": 38},
    },
    # The model config uses the pretrained names (front/left_wrist/right_wrist).
    # The loader maps these positionally to the original LeRobot keys below.
    "video": {
        "cam_head": {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist": {"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        # Keep both spellings: the dataset loader accepts either, while the
        # XPolicyLab GR00T adapter emits human.task_description at inference.
        "human.task_description": {"original_key": "task_index"},
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def _is_deprecated(path: Path, task_root: Path) -> bool:
    return any("deprecated" in part.casefold() for part in path.relative_to(task_root).parts)


def _natural_key(path: Path, root: Path) -> tuple[object, ...]:
    text = path.relative_to(root).as_posix().casefold()
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text))


def _check_raw_episode(path: Path) -> tuple[int, bool]:
    match = RAW_EPISODE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected raw episode filename: {path}")
    with h5py.File(path, "r") as h5:
        required = ("observations/images/main", "observations/qpos", "action")
        missing = [key for key in required if key not in h5]
        if missing:
            raise ValueError(f"{path}: missing datasets {missing}")
        main = h5["observations/images/main"]
        if main.ndim != 4 or tuple(main.shape[1:]) != IMAGE_SHAPE or main.dtype != "uint8":
            raise ValueError(
                f"{path}: main must be (T,384,384,3) uint8, got {main.shape}/{main.dtype}"
            )
        frames = int(main.shape[0])
        if frames < 1:
            raise ValueError(f"{path}: empty episode")
        for key in ("observations/qpos", "action"):
            ds = h5[key]
            if ds.shape != (frames, RAW_DIM) or ds.dtype.kind not in "fiu":
                raise ValueError(f"{path}: /{key} must be (T,50) numeric, got {ds.shape}/{ds.dtype}")
            # Check endpoints without loading the 50-D trajectory into memory.
            for index in (0, frames - 1):
                if not np.isfinite(ds[index]).all():
                    raise ValueError(f"{path}: /{key} contains NaN/Inf")
        wrist_keys = ("observations/images/left_hand", "observations/images/right_hand")
        present = tuple(key in h5 for key in wrist_keys)
        if present[0] != present[1]:
            raise ValueError(f"{path}: left/right wrist camera presence differs")
        if present[0]:
            for key in wrist_keys:
                ds = h5[key]
                if tuple(ds.shape) != tuple(main.shape) or ds.dtype != "uint8":
                    raise ValueError(f"{path}: {key} does not match main camera shape/dtype")
    return frames, present[0]


def audit_raw(source: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise FileNotFoundError(f"raw source does not exist: {source}")
    manifest_path = source / "DATASET_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"filtered raw source is missing {manifest_path.name}")
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != RAW_MANIFEST_SHA256:
        raise ValueError(
            f"raw dataset manifest hash mismatch: {manifest_sha256} != {RAW_MANIFEST_SHA256}"
        )
    raw_manifest = _read_json(manifest_path)
    expected_manifest_fields = {
        "purpose": "active_raw_only",
        "active_task_count": 12,
        "active_episode_count": 1903,
        "active_total_bytes": RAW_TOTAL_BYTES,
        "excluded_deprecated_hdf5": 100,
        "inventory_sha256": RAW_INVENTORY_SHA256,
    }
    for field, expected_value in expected_manifest_fields.items():
        if raw_manifest.get(field) != expected_value:
            raise ValueError(
                f"raw dataset manifest {field}={raw_manifest.get(field)!r}, "
                f"expected {expected_value!r}"
            )
    if raw_manifest.get("tasks") != EXPECTED_TASK_EPISODES:
        raise ValueError("raw dataset manifest task counts do not match the benchmark contract")
    active: list[dict[str, Any]] = []
    deprecated: list[str] = []
    task_counts: Counter[str] = Counter()
    task_frames: Counter[str] = Counter()
    real_wrist = 0

    for task in TASK_INSTRUCTIONS:
        task_root = source / task
        if not task_root.is_dir():
            raise FileNotFoundError(f"missing task directory: {task_root}")
        candidates = sorted(task_root.rglob("episode_*.hdf5"), key=lambda p: _natural_key(p, task_root))
        for path in candidates:
            if _is_deprecated(path, task_root):
                deprecated.append(path.relative_to(source).as_posix())
                continue
            frames, has_wrist = _check_raw_episode(path)
            match = RAW_EPISODE_RE.fullmatch(path.name)
            assert match is not None
            active.append(
                {
                    "task": task,
                    "source_relative_path": path.relative_to(source).as_posix(),
                    "source_episode_index": int(match.group(1)),
                    "frames": frames,
                    "has_real_wrist_cameras": has_wrist,
                }
            )
            task_counts[task] += 1
            task_frames[task] += frames
            real_wrist += int(has_wrist)

    unknown_top_dirs = []
    for child in source.iterdir():
        if child.is_dir() and not child.name.startswith(".") and child.name not in TASK_INSTRUCTIONS:
            unknown_top_dirs.append(child.name)
    if unknown_top_dirs:
        raise ValueError(f"unexpected top-level raw task directories: {unknown_top_dirs}")

    if len(deprecated) != EXPECTED_DEPRECATED:
        raise ValueError(
            f"expected exactly {EXPECTED_DEPRECATED} deprecated episodes, found {len(deprecated)}"
        )
    if dict(task_counts) != EXPECTED_TASK_EPISODES:
        raise ValueError(f"active episode counts differ: {dict(task_counts)}")
    if dict(task_frames) != EXPECTED_TASK_FRAMES:
        raise ValueError(f"active frame counts differ: {dict(task_frames)}")
    if len(active) != 1903 or sum(item["frames"] for item in active) != 510546:
        raise ValueError("active raw totals are not 1903 episodes / 510546 frames")
    if real_wrist != 900:
        raise ValueError(f"expected 900 episodes with real wrist cameras, found {real_wrist}")

    manifest_files = raw_manifest.get("files")
    if not isinstance(manifest_files, list) or len(manifest_files) != len(active):
        raise ValueError("raw dataset manifest file inventory count is invalid")
    inventory_by_path = {row.get("path"): row for row in manifest_files}
    active_by_path = {row["source_relative_path"]: row for row in active}
    if None in inventory_by_path or set(inventory_by_path) != set(active_by_path):
        raise ValueError("raw dataset manifest file paths differ from the active HDF5 view")
    actual_total_bytes = 0
    for relative_path in active_by_path:
        actual_size = (source / relative_path).stat().st_size
        recorded_size = int(inventory_by_path[relative_path].get("size_bytes", -1))
        if actual_size != recorded_size:
            raise ValueError(
                f"raw file size mismatch for {relative_path}: {actual_size} != {recorded_size}"
            )
        actual_total_bytes += actual_size
    if actual_total_bytes != RAW_TOTAL_BYTES:
        raise ValueError(f"raw active byte total mismatch: {actual_total_bytes} != {RAW_TOTAL_BYTES}")

    return {
        "source_root": str(source),
        "total_episodes": len(active),
        "total_frames": sum(item["frames"] for item in active),
        "total_tasks": len(TASK_INSTRUCTIONS),
        "fps": FPS,
        "dataset_manifest_sha256": manifest_sha256,
        "inventory_sha256": raw_manifest["inventory_sha256"],
        "active_total_bytes": actual_total_bytes,
        "source_excluded_deprecated_hdf5": raw_manifest["excluded_deprecated_hdf5"],
        "deprecated_excluded_count": len(deprecated),
        "deprecated_excluded_paths": deprecated,
        "real_wrist_episodes": real_wrist,
        "task_episode_counts": dict(task_counts),
        "task_frame_counts": dict(task_frames),
        "active_episodes": active,
    }


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_cache(cache: Path, raw_audit: dict[str, Any]) -> dict[str, Any]:
    if not cache.is_dir():
        raise FileNotFoundError(f"audited v2.1 cache does not exist: {cache}")
    meta = cache / "meta"
    info = _read_json(meta / "info.json")
    expected = raw_audit["total_episodes"], raw_audit["total_frames"]
    actual = info.get("total_episodes"), info.get("total_frames")
    if info.get("codebase_version") != "v2.1" or actual != expected or info.get("fps") != FPS:
        raise ValueError(f"cache info mismatch: version/totals/fps={info.get('codebase_version')}/{actual}/{info.get('fps')}")
    features = info.get("features", {})
    for key in ("observation.state", "action"):
        if features.get(key, {}).get("shape") != [POLICY_DIM]:
            raise ValueError(f"cache feature {key} is not 38-D")
    for required in (
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
        "meta/stats.json",
        "meta/relative_stats.json",
    ):
        if not (cache / required).is_file():
            raise FileNotFoundError(f"cache missing {required}")

    episodes = _read_jsonl(meta / "episodes.jsonl")
    if len(episodes) != raw_audit["total_episodes"]:
        raise ValueError(f"cache episode count mismatch: {len(episodes)}")
    if [int(row["episode_index"]) for row in episodes] != list(range(len(episodes))):
        raise ValueError("cache episode indices are not contiguous")
    if sum(int(row["length"]) for row in episodes) != raw_audit["total_frames"]:
        raise ValueError("cache episode frame total mismatch")

    tasks = _read_jsonl(meta / "tasks.jsonl")
    task_texts = [row.get("task") for row in sorted(tasks, key=lambda row: int(row["task_index"]))]
    accepted_instruction_maps = (TASK_INSTRUCTIONS, TEMPLATE_TASK_INSTRUCTIONS)
    matched_instructions = next(
        (
            instructions
            for instructions in accepted_instruction_maps
            if set(task_texts) == set(instructions.values())
            and len(task_texts) == len(instructions)
        ),
        None,
    )
    if matched_instructions is None:
        raise ValueError("cache task instructions do not match the active EgoVLA task set")
    if any("deprecated" in str(text).casefold() for text in task_texts):
        raise ValueError("cache task metadata contains a deprecated task")

    # Check task distribution through the legacy episode metadata.
    by_instruction = Counter(row["tasks"][0] for row in episodes)
    expected_by_instruction = Counter(
        {
            matched_instructions[task]: int(count)
            for task, count in raw_audit["task_episode_counts"].items()
        }
    )
    if by_instruction != expected_by_instruction:
        raise ValueError(f"cache task distribution mismatch: {by_instruction}")

    modality = _read_json(meta / "modality.json")
    # The source cache has one annotation spelling.  The target will add the
    # adapter-compatible alias, so only verify the structural groups here.
    for modality_name in ("state", "action"):
        groups = modality.get(modality_name, {})
        for name, bounds in GR00T_MODALITY[modality_name].items():
            if groups.get(name) != bounds:
                raise ValueError(f"cache {modality_name}.{name} range mismatch: {groups.get(name)}")
    if list(modality.get("video", {})) != list(GR00T_MODALITY["video"]):
        raise ValueError("cache video modality keys/order mismatch")

    source_manifest = meta / "xpolicylab_source_conversion.json"
    if not source_manifest.is_file():
        raise FileNotFoundError("cache is missing its source episode/action provenance manifest")
    manifest = _read_json(source_manifest)
    if manifest.get("include_deprecated") is True:
        raise ValueError("cache was produced with deprecated episodes enabled")
    if manifest.get("raw_commanded_action_feature") != "provenance.raw_commanded_action":
        raise ValueError("cache does not expose the raw commanded action provenance column")
    if manifest.get("policy_joint_indices_in_raw_50d") != list(POLICY_JOINT_INDICES):
        raise ValueError("cache 50-D to 38-D action index map does not match XPolicyLab")
    if features.get("provenance.raw_commanded_action", {}).get("shape") != [POLICY_DIM]:
        raise ValueError("cache raw commanded action provenance is not 38-D")

    manifest_episodes = manifest.get("episodes")
    if not isinstance(manifest_episodes, list) or len(manifest_episodes) != len(episodes):
        raise ValueError("cache source manifest episode count does not match metadata")
    if [int(row.get("dataset_episode_index", -1)) for row in manifest_episodes] != list(
        range(len(episodes))
    ):
        raise ValueError("cache source manifest episode indices are not contiguous")

    raw_by_path = {row["source_relative_path"]: row for row in raw_audit["active_episodes"]}
    manifest_by_path = {row.get("source_relative_path"): row for row in manifest_episodes}
    if None in manifest_by_path or set(manifest_by_path) != set(raw_by_path):
        missing = sorted(set(raw_by_path) - set(manifest_by_path))[:10]
        extra = sorted(set(manifest_by_path) - set(raw_by_path))[:10]
        raise ValueError(f"cache/raw source episode set mismatch: missing={missing}, extra={extra}")
    for relative_path, raw_row in raw_by_path.items():
        cached_row = manifest_by_path[relative_path]
        for field in ("task", "source_episode_index", "frames", "has_real_wrist_cameras"):
            if cached_row.get(field) != raw_row[field]:
                raise ValueError(
                    f"cache source manifest mismatch for {relative_path} field {field}: "
                    f"{cached_row.get(field)!r} != {raw_row[field]!r}"
                )

    return {
        "codebase_version": info.get("codebase_version"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "total_tasks": info.get("total_tasks"),
        "fps": info.get("fps"),
        "source_cache": str(cache),
        "template_training_action_source": manifest.get("training_action_source"),
        "template_episode_map": manifest_episodes,
    }


def _copy_tree(source: Path, destination: Path) -> str:
    """Copy metadata, rewrite parquet later, and hard-link immutable videos."""
    destination.mkdir(parents=True, exist_ok=False)
    modes: set[str] = set()
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        dest_root = destination / relative
        dest_root.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (dest_root / directory).mkdir(parents=True, exist_ok=True)
        for name in files:
            src = root_path / name
            dst = dest_root / name
            if relative.parts and relative.parts[0] == "meta":
                shutil.copy2(src, dst)
                modes.add("copy-meta")
                continue
            if relative.parts and relative.parts[0] == "data" and src.suffix == ".parquet":
                modes.add("rewrite-parquet")
                continue
            try:
                os.link(src, dst)
                modes.add("hardlink-payload")
            except OSError:
                shutil.copy2(src, dst)
                modes.add("copy-payload")
    return "+".join(sorted(modes))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _rewrite_prompt_metadata(staging: Path) -> None:
    corrected = TEMPLATE_TASK_INSTRUCTIONS["Insert-And-Unload-Cans"]
    canonical = TASK_INSTRUCTIONS["Insert-And-Unload-Cans"]
    tasks_path = staging / "meta" / "tasks.jsonl"
    task_rows = _read_jsonl(tasks_path)
    replacements = 0
    for row in task_rows:
        if row.get("task") == corrected:
            row["task"] = canonical
            replacements += 1
    if replacements != 1:
        raise ValueError(f"expected one corrected prompt in tasks.jsonl, found {replacements}")
    _write_jsonl(tasks_path, task_rows)

    episodes_path = staging / "meta" / "episodes.jsonl"
    episode_rows = _read_jsonl(episodes_path)
    episode_replacements = 0
    for row in episode_rows:
        row_tasks = row.get("tasks", [])
        rewritten = [canonical if task == corrected else task for task in row_tasks]
        episode_replacements += sum(task == corrected for task in row_tasks)
        row["tasks"] = rewritten
    if episode_replacements != EXPECTED_TASK_EPISODES["Insert-And-Unload-Cans"]:
        raise ValueError(
            "unexpected Insert-And-Unload-Cans episode prompt count: "
            f"{episode_replacements}"
        )
    _write_jsonl(episodes_path, episode_rows)


def _column_matrix(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise KeyError(f"parquet is missing required column {name!r}")
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != POLICY_DIM:
        raise ValueError(f"parquet column {name!r} must have shape [T,{POLICY_DIM}], got {values.shape}")
    return values


def _episode_parquet(root: Path, episode_index: int) -> Path:
    matches = list(root.glob(f"data/*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one parquet for episode {episode_index}, found {len(matches)} under {root}"
        )
    return matches[0]


def _rewrite_actions(
    cache: Path,
    staging: Path,
    raw_source: Path,
    episode_map: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rewrite every action from raw ``/action[t]`` and prove its provenance."""
    action_digest = hashlib.sha256()
    state_digest = hashlib.sha256()
    old_next_state_episodes = 0
    absolute_error_sum = 0.0
    absolute_error_count = 0
    max_absolute_error = 0.0

    for item in episode_map:
        episode_index = int(item["dataset_episode_index"])
        relative_path = str(item["source_relative_path"])
        raw_path = raw_source / relative_path
        source_parquet = _episode_parquet(cache, episode_index)
        destination_parquet = staging / source_parquet.relative_to(cache)
        destination_parquet.parent.mkdir(parents=True, exist_ok=True)

        table = pq.read_table(source_parquet)
        frames = table.num_rows
        if frames != int(item["frames"]):
            raise ValueError(
                f"episode {episode_index} row count {frames} != manifest frames {item['frames']}"
            )
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if not np.all(episode_ids == episode_index) or not np.array_equal(
            frame_ids, np.arange(frames, dtype=np.int64)
        ):
            raise ValueError(f"episode/frame indices are corrupt in {source_parquet}")

        with h5py.File(raw_path, "r") as h5:
            raw_action = np.ascontiguousarray(
                np.asarray(h5["action"][:], dtype=np.float32)[:, POLICY_JOINT_INDICES]
            )
            raw_state = np.ascontiguousarray(
                np.asarray(h5["observations/qpos"][:], dtype=np.float32)[:, POLICY_JOINT_INDICES]
            )
        if raw_action.shape != (frames, POLICY_DIM) or raw_state.shape != (frames, POLICY_DIM):
            raise ValueError(f"raw action/state shape mismatch for {raw_path}")
        if not np.isfinite(raw_action).all() or not np.isfinite(raw_state).all():
            raise ValueError(f"raw action/state contains NaN or Inf: {raw_path}")

        parquet_state = _column_matrix(table, "observation.state")
        provenance_action = _column_matrix(table, "provenance.raw_commanded_action")
        previous_training_action = _column_matrix(table, "action")
        if not np.array_equal(parquet_state, raw_state):
            raise ValueError(f"template state is not same-timestep raw qpos for {raw_path}")
        if not np.array_equal(provenance_action, raw_action):
            raise ValueError(f"template raw-action provenance differs from HDF5 /action for {raw_path}")

        next_state = np.concatenate((raw_state[1:], raw_state[-1:]), axis=0)
        old_next_state_episodes += int(np.array_equal(previous_training_action, next_state))
        error = np.abs(previous_training_action - raw_action)
        absolute_error_sum += float(error.sum(dtype=np.float64))
        absolute_error_count += int(error.size)
        max_absolute_error = max(max_absolute_error, float(error.max(initial=0.0)))

        action_field = table.schema.field("action")
        action_column = pa.array(raw_action.tolist(), type=action_field.type)
        rewritten = table.set_column(table.schema.get_field_index("action"), action_field, action_column)
        metadata = pq.read_metadata(source_parquet)
        compression = metadata.row_group(0).column(0).compression.lower()
        if compression == "uncompressed":
            compression = None
        pq.write_table(rewritten, destination_parquet, compression=compression)

        persisted = _column_matrix(pq.read_table(destination_parquet, columns=["action"]), "action")
        if not np.array_equal(persisted, raw_action):
            raise ValueError(f"persisted action verification failed for {destination_parquet}")
        action_digest.update(raw_action.astype("<f4", copy=False).tobytes(order="C"))
        state_digest.update(raw_state.astype("<f4", copy=False).tobytes(order="C"))

        if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episode_map):
            print(f"ACTION_REWRITE progress={episode_index + 1}/{len(episode_map)}", flush=True)

    return {
        "episodes_rewritten": len(episode_map),
        "frames_rewritten": sum(int(item["frames"]) for item in episode_map),
        "training_action_source": "raw HDF5 /action at the same timestep",
        "training_action_formula": "action38[t] = hdf5['/action'][t, policy_joint_indices_in_raw_50d]",
        "next_observed_state_used_as_action": False,
        "template_episodes_equal_to_next_observed_qpos": old_next_state_episodes,
        "template_vs_raw_action_mean_absolute_error": (
            absolute_error_sum / absolute_error_count if absolute_error_count else 0.0
        ),
        "template_vs_raw_action_max_absolute_error": max_absolute_error,
        "raw_action38_stream_sha256": action_digest.hexdigest(),
        "raw_state38_stream_sha256": state_digest.hexdigest(),
    }


def _regenerate_statistics(staging: Path, gr00t_root: Path, modality_config: Path) -> None:
    stats_script = gr00t_root / "gr00t" / "data" / "stats.py"
    for required in (stats_script, modality_config):
        if not required.is_file():
            raise FileNotFoundError(f"missing statistics dependency: {required}")
    for stale_name in (
        "stats.json",
        "relative_stats.json",
        "episodes_stats.jsonl",
        "source_v3_stats.json",
    ):
        (staging / "meta" / stale_name).unlink(missing_ok=True)
    env = os.environ.copy()
    python_path = [str(gr00t_root), str(MODEL_ROOT)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(
        [
            sys.executable,
            str(stats_script),
            "--dataset-path",
            str(staging),
            "--embodiment-tag",
            "NEW_EMBODIMENT",
            "--modality-config-path",
            str(modality_config),
        ],
        check=True,
        cwd=gr00t_root,
        env=env,
    )
    for name in ("stats.json", "relative_stats.json"):
        if not (staging / "meta" / name).is_file():
            raise FileNotFoundError(f"statistics generation did not create meta/{name}")


def validate_output(output: Path, raw_audit: dict[str, Any]) -> dict[str, Any]:
    cache_audit = validate_cache(output, raw_audit)
    manifest_path = output / "meta" / "egovla_groot_conversion.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("converted dataset is missing egovla_groot_conversion.json")
    manifest = _read_json(manifest_path)
    action_audit = manifest.get("raw_action_rewrite", {})
    if action_audit.get("next_observed_state_used_as_action") is not False:
        raise ValueError("converted dataset does not explicitly forbid next-state action labels")
    if action_audit.get("training_action_source") != "raw HDF5 /action at the same timestep":
        raise ValueError("converted dataset is not bound to raw HDF5 /action")
    if action_audit.get("episodes_rewritten") != raw_audit["total_episodes"]:
        raise ValueError("converted dataset raw-action episode count is incomplete")
    source_manifest = _read_json(output / "meta" / "xpolicylab_source_conversion.json")
    if source_manifest.get("training_action_source") != "raw HDF5 /action at the same timestep":
        raise ValueError("source conversion manifest still points at a derived/next-state action")
    task_rows = sorted(
        _read_jsonl(output / "meta" / "tasks.jsonl"), key=lambda row: int(row["task_index"])
    )
    if [row.get("task") for row in task_rows] != list(TASK_INSTRUCTIONS.values()):
        raise ValueError("published task prompts do not exactly match the benchmark registry")
    expected_hashes = manifest.get("dataset_meta_sha256", {})
    for name in ("info.json", "modality.json", "tasks.jsonl", "stats.json", "relative_stats.json"):
        actual = _sha256(output / "meta" / name)
        if expected_hashes.get(name) != actual:
            raise ValueError(f"converted dataset metadata hash mismatch for {name}")
    return cache_audit


def materialize(
    cache: Path,
    output: Path,
    raw_source: Path,
    raw_audit: dict[str, Any],
    cache_audit: dict[str, Any],
    gr00t_root: Path,
    modality_config: Path,
    force: bool,
) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not force:
            # Idempotent reruns are safe when the existing output is already a
            # validated conversion; do not overwrite arbitrary user data.
            try:
                existing = validate_output(output, raw_audit)
                print(f"ALREADY_VALID output={output} episodes={existing['total_episodes']}")
                return
            except Exception as exc:
                raise FileExistsError(
                    f"refusing to overwrite existing output {output}; use --force only for this exact target"
                ) from exc
        backup = output.with_name(
            output.name + f".backup-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        )
        output.rename(backup)
        print(f"BACKUP old_output={backup}")

    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    try:
        copy_mode = _copy_tree(cache, staging)
        action_audit = _rewrite_actions(
            cache,
            staging,
            raw_source,
            cache_audit["template_episode_map"],
        )
        # Never mutate a hard-linked source metadata file.  _copy_tree copies
        # all metadata independently, so these writes affect only the target.
        _write_json(staging / "meta" / "modality.json", GR00T_MODALITY)
        _rewrite_prompt_metadata(staging)
        source_manifest_path = staging / "meta" / "xpolicylab_source_conversion.json"
        template_source_manifest_path = (
            staging / "meta" / "template_xpolicylab_source_conversion.json"
        )
        shutil.copy2(source_manifest_path, template_source_manifest_path)
        source_manifest = _read_json(source_manifest_path)
        source_manifest.update(
            {
                "converter": Path(__file__).name,
                "converter_version": "2.0.0",
                "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "source_root": str(raw_source),
                "output_root": str(output),
                "include_deprecated": False,
                "training_action_source": "raw HDF5 /action at the same timestep",
                "training_action_formula": action_audit["training_action_formula"],
                "next_observed_state_used_as_action": False,
                "raw_action38_stream_sha256": action_audit["raw_action38_stream_sha256"],
                "instructions": TASK_INSTRUCTIONS,
            }
        )
        _write_json(source_manifest_path, source_manifest)
        _regenerate_statistics(staging, gr00t_root, modality_config)
        cache_summary = {
            key: value for key, value in cache_audit.items() if key != "template_episode_map"
        }
        manifest = {
            "converter": Path(__file__).name,
            "converter_version": "2.0.0",
            "converted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_audit": raw_audit,
            "audited_v21_template": cache_summary,
            "output_root": str(output),
            "copy_mode": copy_mode,
            "deprecated_policy": "source is raw_remove_deprecated; deprecated count must be zero",
            "raw_action_rewrite": action_audit,
            "action_type": "joint",
            "action_dim": POLICY_DIM,
            "action_horizon": 16,
            "action_representation": {
                "left_arm": "relative to same-timestep state (processor transform)",
                "right_arm": "relative to same-timestep state (processor transform)",
                "left_hand": "absolute",
                "right_hand": "absolute",
            },
            "policy_joint_indices_in_raw_50d": list(POLICY_JOINT_INDICES),
            "state_action_layout": GR00T_MODALITY["state"],
            "language_modality": "annotation.human.task_description -> task_index",
            "prompts": TASK_INSTRUCTIONS,
            "video_modality": GR00T_MODALITY["video"],
            "camera_contract": {
                "color_order": "RGB",
                "raw_shape_hwc": list(IMAGE_SHAPE),
                "real_wrist_episodes": raw_audit["real_wrist_episodes"],
                "black_wrist_episodes": raw_audit["total_episodes"]
                - raw_audit["real_wrist_episodes"],
                "missing_wrist_policy": "uint8 zero frames for both wrists; never duplicate head",
            },
            "statistics_generator": str(gr00t_root / "gr00t" / "data" / "stats.py"),
            "modality_config": str(modality_config),
            "dataset_meta_sha256": {
                name: _sha256(staging / "meta" / name)
                for name in (
                    "info.json",
                    "modality.json",
                    "tasks.jsonl",
                    "stats.json",
                    "relative_stats.json",
                    "xpolicylab_source_conversion.json",
                )
            },
        }
        _write_json(staging / "meta" / "egovla_groot_conversion.json", manifest)
        # A final validation reads the staged metadata before it becomes visible.
        validate_output(staging, raw_audit)
        staging.replace(output)
        print(
            f"DONE output={output} episodes={raw_audit['total_episodes']} "
            f"frames={raw_audit['total_frames']} deprecated_excluded={raw_audit['deprecated_excluded_count']} "
            f"copy_mode={copy_mode}"
        )
    except Exception:
        print(f"FAILED; partial staging retained at {staging}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--source-v21", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gr00t-root", type=Path, default=DEFAULT_GR00T_ROOT)
    parser.add_argument("--modality-config", type=Path, default=DEFAULT_MODALITY_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="audit only; do not materialize data")
    parser.add_argument("--force", action="store_true", help="backup and replace the exact output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"AUDIT raw={args.source_raw}")
    raw_audit = audit_raw(args.source_raw)
    print(
        f"RAW_OK active_episodes={raw_audit['total_episodes']} "
        f"frames={raw_audit['total_frames']} deprecated_excluded={raw_audit['deprecated_excluded_count']} "
        f"real_wrist={raw_audit['real_wrist_episodes']}"
    )
    cache_audit = validate_cache(args.source_v21, raw_audit)
    print(f"CACHE_OK path={args.source_v21} version={cache_audit['codebase_version']}")
    if args.dry_run:
        return
    materialize(
        args.source_v21,
        args.output,
        args.source_raw,
        raw_audit,
        cache_audit,
        args.gr00t_root,
        args.modality_config,
        args.force,
    )


if __name__ == "__main__":
    main()
