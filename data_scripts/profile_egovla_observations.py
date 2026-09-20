#!/usr/bin/env python3
"""Create the checked EgoVLA train/eval observation contract.

The converted LeRobot dataset is already audited against the raw HDF5 source.
This tool turns its immutable conversion metadata into the small sidecar used
by the training launcher, while independently checking camera layout, task
distribution, RGB resolution, and the single three-view task.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


IMAGE_SHAPE = [384, 384, 3]
VIDEO_KEYS = {
    "cam_head": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}
EXPECTED_TASK_COUNTS = {
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
REAL_WRIST_TASK = "Insert-And-Unload-Cans"
EXPECTED_ACTION_SHA256 = "96dfbce561c21dd413845c50b05da6ce77f1a2b0e8dd2312bc25f85de4ebdfd2"
EXPECTED_STATE_SHA256 = "78f21e4f2376c85d8deed8795ea190c87ee29b941e84834e8efe7814b4e80a7e"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_row(job: tuple[Path, Path]) -> tuple[str, int, str]:
    dataset, path = job
    return path.relative_to(dataset).as_posix(), path.stat().st_size, sha256(path)


def video_inventory(dataset: Path, feature_key: str, workers: int) -> tuple[int, int, str]:
    paths = sorted(dataset.glob(f"videos/chunk-*/{feature_key}/episode_*.mp4"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(payload_row, ((dataset, path) for path in paths)))
    digest = hashlib.sha256()
    for relative_path, size_bytes, file_sha256 in rows:
        digest.update(f"{relative_path}\0{size_bytes}\0{file_sha256}\n".encode())
    return len(rows), sum(row[1] for row in rows), digest.hexdigest()


def episode_parquet(dataset: Path, episode_index: int) -> Path:
    matches = list(dataset.glob(f"data/*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"episode {episode_index}: expected one parquet, found {len(matches)}"
        )
    return matches[0]


def matrix(table, name: str) -> np.ndarray:
    values = np.asarray(table[name].to_pylist(), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 38:
        raise ValueError(f"{name}: expected [T,38], got {values.shape}")
    return np.ascontiguousarray(values)


def feature_shape(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if shape is not None:
        return list(shape)
    video_info = feature.get("info", {})
    if all(name in video_info for name in ("video.height", "video.width", "video.channels")):
        return [
            int(video_info["video.height"]),
            int(video_info["video.width"]),
            int(video_info["video.channels"]),
        ]
    return None


def build_profile(dataset: Path, video_workers: int) -> dict[str, Any]:
    dataset = dataset.resolve()
    meta = dataset / "meta"
    conversion_path = meta / "egovla_groot_conversion.json"
    source_path = meta / "xpolicylab_source_conversion.json"
    info_path = meta / "info.json"
    modality_path = meta / "modality.json"
    video_audit_path = meta / "egovla_v21_video_audit.json"
    for required in (
        conversion_path,
        source_path,
        info_path,
        modality_path,
        video_audit_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    conversion = read_json(conversion_path)
    source = read_json(source_path)
    info = read_json(info_path)
    modality = read_json(modality_path)
    video_audit = read_json(video_audit_path)

    if conversion.get("modality_config_sha256") != (
        "fcdadfac0d6a94aacdd33ae07f5b87ef18e01b55926c50f3a42980ff454b2f86"
    ):
        raise ValueError("conversion used an unexpected EgoVLA modality config")

    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"expected LeRobot v2.1, got {info.get('codebase_version')!r}")
    if (info.get("total_episodes"), info.get("total_frames"), info.get("fps")) != (
        1903,
        510546,
        30,
    ):
        raise ValueError("unexpected EgoVLA episode/frame/fps totals")

    camera_contract = conversion.get("camera_contract", {})
    if camera_contract != {
        "color_order": "RGB",
        "raw_shape_hwc": IMAGE_SHAPE,
        "real_wrist_episodes": 900,
        "black_wrist_episodes": 1003,
        "missing_wrist_policy": "uint8 zero frames for both wrists; never duplicate head",
    }:
        raise ValueError(f"unexpected camera contract: {camera_contract}")

    features = info.get("features", {})
    video_modality = modality.get("video", {})
    if list(video_modality) != list(VIDEO_KEYS):
        raise ValueError(f"unexpected video modality order: {list(video_modality)}")
    for modality_key, feature_key in VIDEO_KEYS.items():
        if video_modality[modality_key].get("original_key") != feature_key:
            raise ValueError(f"{modality_key} does not map to {feature_key}")
        feature = features.get(feature_key, {})
        if feature.get("dtype") != "video" or feature_shape(feature) != IMAGE_SHAPE:
            raise ValueError(f"unexpected {feature_key} feature: {feature}")

    episodes = source.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1903:
        raise ValueError("source provenance must contain exactly 1903 episodes")
    counts: Counter[str] = Counter()
    wrist_values: dict[str, set[bool]] = defaultdict(set)
    for expected_index, episode in enumerate(episodes):
        if int(episode.get("dataset_episode_index", -1)) != expected_index:
            raise ValueError("source provenance episode indices are not contiguous")
        task = str(episode.get("task"))
        counts[task] += 1
        wrist_values[task].add(bool(episode.get("has_real_wrist_cameras")))
    if dict(counts) != EXPECTED_TASK_COUNTS:
        raise ValueError(f"unexpected task distribution: {dict(counts)}")
    expected_wrist_values = {task: {task == REAL_WRIST_TASK} for task in EXPECTED_TASK_COUNTS}
    if dict(wrist_values) != expected_wrist_values:
        raise ValueError(f"task camera provenance mismatch: {dict(wrist_values)}")

    action_digest = hashlib.sha256()
    state_digest = hashlib.sha256()
    payload_frames = 0
    for expected_index, episode in enumerate(episodes):
        table = pq.read_table(
            episode_parquet(dataset, expected_index),
            columns=[
                "observation.state",
                "action",
                "provenance.raw_commanded_action",
                "episode_index",
                "frame_index",
            ],
        )
        expected_frames = int(episode["frames"])
        if table.num_rows != expected_frames:
            raise ValueError(
                f"episode {expected_index}: rows={table.num_rows}, expected={expected_frames}"
            )
        episode_ids = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_ids = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        if not np.all(episode_ids == expected_index) or not np.array_equal(
            frame_ids, np.arange(expected_frames, dtype=np.int64)
        ):
            raise ValueError(f"episode {expected_index}: corrupt episode/frame indices")
        action = matrix(table, "action")
        provenance = matrix(table, "provenance.raw_commanded_action")
        state = matrix(table, "observation.state")
        if not np.array_equal(action, provenance):
            raise ValueError(f"episode {expected_index}: action differs from raw provenance")
        action_digest.update(action.astype("<f4", copy=False).tobytes(order="C"))
        state_digest.update(state.astype("<f4", copy=False).tobytes(order="C"))
        payload_frames += expected_frames
    action_sha256 = action_digest.hexdigest()
    state_sha256 = state_digest.hexdigest()
    if (
        payload_frames != 510546
        or action_sha256 != EXPECTED_ACTION_SHA256
        or state_sha256 != EXPECTED_STATE_SHA256
    ):
        raise ValueError(
            "parquet payload identity mismatch: "
            f"frames={payload_frames}, action={action_sha256}, state={state_sha256}"
        )

    if video_audit.get("schema") != "egovla-v21-video-audit-v1":
        raise ValueError("missing or incompatible v2.1 video payload audit")
    audited_cameras = video_audit.get("cameras", {})
    video_payload: dict[str, Any] = {}
    for feature_key in VIDEO_KEYS.values():
        videos, bytes_total, inventory_sha256 = video_inventory(dataset, feature_key, video_workers)
        audited = audited_cameras.get(feature_key, {})
        if (
            videos != 1903
            or inventory_sha256 != audited.get("inventory_sha256")
            or bytes_total != audited.get("bytes")
            or audited.get("frames") != 510546
        ):
            raise ValueError(f"{feature_key}: video payload inventory differs from full audit")
        video_payload[feature_key] = {
            "videos": videos,
            "bytes": bytes_total,
            "inventory_sha256": inventory_sha256,
        }
    for wrist_key in (
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ):
        audited = audited_cameras[wrist_key]
        if audited.get("verified_black_frames") != 194547 or audited.get("verified_black_max") != 0:
            raise ValueError(f"{wrist_key}: full black-frame audit is not valid")
    rgb_alignment = video_audit.get("rgb_alignment", {})
    if set(rgb_alignment) != set(EXPECTED_TASK_COUNTS):
        raise ValueError("RGB alignment audit does not cover the exact 12-task set")
    head_key = "observation.images.cam_high"
    wrist_keys = {
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    for task, camera_results in rgb_alignment.items():
        expected_cameras = {head_key}
        if task == REAL_WRIST_TASK:
            expected_cameras |= wrist_keys
        if set(camera_results) != expected_cameras:
            raise ValueError(f"{task}: unexpected RGB audit cameras {list(camera_results)}")
        for camera_key, result in camera_results.items():
            direct = result.get("direct_rgb_mae")
            swapped = result.get("channel_swapped_mae")
            if not isinstance(direct, (int, float)) or not isinstance(swapped, (int, float)):
                raise ValueError(f"{task}/{camera_key}: missing RGB audit metrics")
            if not np.isfinite(direct) or not np.isfinite(swapped) or not direct < swapped:
                raise ValueError(f"{task}/{camera_key}: RGB alignment is not better than BGR swap")

    task_camera_masks = {
        f"Humanoid-{task}-v0": [True, task == REAL_WRIST_TASK, task == REAL_WRIST_TASK]
        for task in EXPECTED_TASK_COUNTS
    }
    return {
        "schema": "egovla-observation-profile-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "dataset": str(dataset),
            "conversion_manifest": str(conversion_path.resolve()),
            "conversion_manifest_sha256": sha256(conversion_path),
            "source_manifest_sha256": sha256(source_path),
            "video_audit_sha256": sha256(video_audit_path),
        },
        "color_order": "RGB",
        "camera_layout": "HWC",
        "camera_dtype": "uint8",
        "camera_shapes": {name: IMAGE_SHAPE[:2] for name in VIDEO_KEYS},
        "task_camera_masks": task_camera_masks,
        "missing_wrist_policy": camera_contract["missing_wrist_policy"],
        "payload": {
            "frames": payload_frames,
            "raw_action38_stream_sha256": action_sha256,
            "raw_state38_stream_sha256": state_sha256,
            "videos": video_payload,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(args.dataset, args.video_workers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"OBSERVATION_PROFILE_OK output={args.output}")


if __name__ == "__main__":
    main()
