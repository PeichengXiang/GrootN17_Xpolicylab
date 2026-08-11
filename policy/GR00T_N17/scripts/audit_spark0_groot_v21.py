#!/usr/bin/env python3
"""Independently audit a converted Spark0 GR00T LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd

from XPolicyLab.utils import process_data


TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
EXPECTED_SLICES = {
    "left_arm": (0, 7),
    "left_hand": (7, 27),
    "right_arm": (27, 34),
    "right_hand": (34, 54),
}
PROCESSOR_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")
VIDEO_MAP = {
    "front": ("observation.images.cam_high", "vision/cam_head/colors"),
    "left_wrist": (
        "observation.images.cam_left_wrist",
        "vision/cam_left_wrist/colors",
    ),
    "right_wrist": (
        "observation.images.cam_right_wrist",
        "vision/cam_right_wrist/colors",
    ),
}
STAT_TYPES = ("mean", "std", "min", "max", "q01", "q99")
FPS = 25
HEIGHT = 480
WIDTH = 640
JOINT_DIM = 54
VIDEO_CODEC = "h264"
FULL_EPISODES = 600
FULL_FRAMES = 151410
MARKER_RELATIVE_PATH = Path("meta/spark0_joint54_audit.json")
MARKER_VERSION = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity(dataset: Path) -> dict[str, Any]:
    marker = MARKER_RELATIVE_PATH.as_posix()
    marker_tmp = f"{marker}.tmp"
    paths = [
        path
        for path in sorted(dataset.rglob("*"))
        if path.is_file()
        and path.relative_to(dataset).as_posix() not in {marker, marker_tmp}
    ]

    def identity_entry(path: Path) -> dict[str, Any]:
        relative = path.relative_to(dataset).as_posix()
        return {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as executor:
        entries = list(executor.map(identity_entry, paths))

    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "algorithm": "sha256(paths+sizes+all-file-content)-v2",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(entries),
        "total_bytes": sum(entry["size"] for entry in entries),
    }


def _decode_instruction(value: Any) -> str:
    for item in np.asarray(value).reshape(-1):
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, (bytes, bytearray, memoryview)):
            text = bytes(item).rstrip(b"\0").decode("utf-8")
        else:
            text = str(item)
        if text.strip():
            return text.strip()
    return ""


def _raw_vectors(path: Path) -> tuple[np.ndarray, np.ndarray, str, int]:
    with h5py.File(path, "r") as handle:
        def vector(prefix: str) -> np.ndarray:
            return np.concatenate(
                [
                    handle[f"{prefix}/left_arm_joint_states"][:],
                    handle[f"{prefix}/left_ee_joint_states"][:],
                    handle[f"{prefix}/right_arm_joint_states"][:],
                    handle[f"{prefix}/right_ee_joint_states"][:],
                ],
                axis=1,
            ).astype(np.float32)

        state = vector("state")
        action = vector("action")
        instruction = _decode_instruction(handle["instruction"][()])
        fps = int(handle["additional_info/frequency"][()])
    return state, action, instruction, fps


def _decode_hdf5_frame(path: Path, key: str, index: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        encoded = handle[key][index]
    image = np.asarray(process_data.decode_image_bit(encoded))
    _require(
        image.shape == (HEIGHT, WIDTH, 3) and image.dtype == np.uint8,
        f"Decoded source image has invalid shape/dtype: {path}:{key}[{index}] "
        f"{image.shape}/{image.dtype}",
    )
    return image


def _episode_files(source_root: Path, task: str) -> list[Path]:
    data_root = source_root / task / "tianji_marvin_wuji/data"
    files = sorted(data_root.glob("episode_*.hdf5"))
    files.extend(sorted(data_root.glob("episode_*.h5")))
    unique = list(dict.fromkeys(path.resolve() for path in files))
    _require(len(unique) == 100, f"{task}: expected 100 raw episodes, got {len(unique)}")
    return unique


def _expected_joint_names() -> list[str]:
    return (
        [f"left_arm_joint_{index}" for index in range(7)]
        + [f"left_hand_joint_{index}" for index in range(20)]
        + [f"right_arm_joint_{index}" for index in range(7)]
        + [f"right_hand_joint_{index}" for index in range(20)]
    )


def _check_metadata(
    dataset: Path,
    source_root: Path,
    expected_episodes: int | None,
    require_full: bool,
) -> dict[str, Any]:
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    episodes = _json_lines(dataset / "meta/episodes.jsonl")
    tasks = _json_lines(dataset / "meta/tasks.jsonl")
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))
    manifest_path = dataset / "meta/spark0_source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    episode_count = len(manifest)
    if expected_episodes is not None:
        _require(
            episode_count == expected_episodes,
            f"Manifest has {episode_count} episodes, expected {expected_episodes}",
        )
    _require(episode_count > 0 and episode_count % len(TASKS) == 0, "Uneven task count")
    per_task = episode_count // len(TASKS)
    _require(1 <= per_task <= 100, f"Invalid episodes per task: {per_task}")
    if require_full:
        _require(episode_count == FULL_EPISODES, f"Expected {FULL_EPISODES} episodes")

    _require(info["codebase_version"] == "v2.1", "Dataset is not LeRobot v2.1")
    _require(info["robot_type"] == "tianji_marvin_wuji", "Unexpected robot_type")
    _require(info["fps"] == FPS, f"Expected {FPS} Hz, got {info['fps']}")
    _require(
        info["total_episodes"] == len(episodes) == episode_count,
        "Episode totals disagree",
    )
    _require(info["total_tasks"] == len(tasks) == len(TASKS), "Task totals disagree")
    total_frames = sum(int(item["frames"]) for item in manifest)
    _require(info["total_frames"] == total_frames, "Frame totals disagree")
    if require_full:
        _require(total_frames == FULL_FRAMES, f"Expected {FULL_FRAMES} frames")
    _require(info["total_videos"] == episode_count * len(VIDEO_MAP), "Video total mismatch")
    _require(info["splits"] == {"train": f"0:{episode_count}"}, "Unexpected split")

    features = info["features"]
    expected_names = _expected_joint_names()
    for key in ("observation.state", "action"):
        feature = features[key]
        _require(feature["dtype"] == "float32", f"{key} is not float32")
        _require(feature["shape"] == [JOINT_DIM], f"{key} is not {JOINT_DIM}D")
        _require(feature["names"] == [expected_names], f"{key} names/order mismatch")

    expected_video_features = {value[0] for value in VIDEO_MAP.values()}
    actual_video_features = {
        key for key, value in features.items() if value["dtype"] == "video"
    }
    _require(actual_video_features == expected_video_features, "Video feature set mismatch")
    for key in expected_video_features:
        feature = features[key]
        _require(feature["shape"] == [3, HEIGHT, WIDTH], f"{key} shape mismatch")
        video_info = feature["info"]
        _require(video_info["video.height"] == HEIGHT, f"{key} height mismatch")
        _require(video_info["video.width"] == WIDTH, f"{key} width mismatch")
        _require(video_info["video.fps"] == FPS, f"{key} fps mismatch")
        _require(video_info["video.channels"] == 3, f"{key} channel mismatch")
        _require(video_info["video.codec"] == VIDEO_CODEC, f"{key} codec mismatch")

    expected_ranges = {
        key: {"start": start, "end": end}
        for key, (start, end) in EXPECTED_SLICES.items()
    }
    _require(modality["state"] == expected_ranges, "State modality ranges mismatch")
    _require(modality["action"] == expected_ranges, "Action modality ranges mismatch")
    _require(list(modality["video"]) == list(VIDEO_MAP), "Video modality order mismatch")
    _require(
        modality["annotation"]
        == {"human.task_description": {"original_key": "task_index"}},
        "Language modality mismatch",
    )
    serialized = json.dumps({"info": info, "modality": modality}).lower()
    _require(
        not any(token in serialized for token in ("waist", "torso", "gripper")),
        "Unexpected waist/torso/gripper field",
    )

    _require(
        [item["episode_index"] for item in episodes] == list(range(episode_count)),
        "Episode indices are not contiguous",
    )
    _require(
        [item["length"] for item in episodes]
        == [int(item["frames"]) for item in manifest],
        "Episode lengths disagree with manifest",
    )
    _require(
        [item["tasks"] for item in episodes]
        == [[item["instruction"]] for item in manifest],
        "Episode instructions disagree with manifest",
    )
    _require(
        [item["task_index"] for item in tasks] == list(range(len(TASKS))),
        "Task indices are not contiguous",
    )

    instruction_by_task: dict[str, str] = {}
    for task in TASKS:
        task_items = [item for item in manifest if item["task"] == task]
        _require(len(task_items) == per_task, f"{task}: expected {per_task} manifest items")
        instructions = {str(item["instruction"]) for item in task_items}
        _require(len(instructions) == 1, f"{task}: multiple instructions in raw manifest")
        instruction_by_task[task] = next(iter(instructions))
        expected_sources = _episode_files(source_root, task)[:per_task]
        actual_sources = [Path(item["source"]).resolve() for item in task_items]
        _require(actual_sources == expected_sources, f"{task}: source episode list/order mismatch")
        for item, source in zip(task_items, expected_sources):
            _require(int(item["fps"]) == FPS, f"{source}: manifest FPS mismatch")
            _require(source.is_relative_to(source_root), f"Source escapes root: {source}")

    expected_task_strings = [instruction_by_task[task] for task in TASKS]
    _require([item["task"] for item in tasks] == expected_task_strings, "Task text/order mismatch")

    print(
        "METADATA_OK "
        f"v2.1 episodes={episode_count} frames={total_frames} "
        f"videos={episode_count * len(VIDEO_MAP)} fps={FPS} resolution={HEIGHT}x{WIDTH}",
        flush=True,
    )
    return {
        "info": info,
        "manifest": manifest,
        "tasks": tasks,
        "episode_count": episode_count,
        "frame_count": total_frames,
        "per_task": per_task,
        "manifest_path": manifest_path,
    }


def _indexed_files(dataset: Path, pattern: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(dataset.glob(pattern)):
        episode_index = int(path.stem.rsplit("_", 1)[1])
        _require(episode_index not in result, f"Duplicate episode file: {path}")
        result[episode_index] = path
    return result


def _check_parquet(dataset: Path, context: dict[str, Any]) -> None:
    manifest = context["manifest"]
    tasks = context["tasks"]
    parquet = _indexed_files(dataset, "data/chunk-*/episode_*.parquet")
    _require(len(parquet) == len(manifest), "Parquet episode count mismatch")
    task_indices = {item["task"]: int(item["task_index"]) for item in tasks}
    next_global_index = 0

    for episode_index, item in enumerate(manifest):
        _require(episode_index in parquet, f"Missing Parquet episode {episode_index}")
        frame = pd.read_parquet(parquet[episode_index])
        source = Path(item["source"])
        expected_state, expected_action, instruction, source_fps = _raw_vectors(source)
        actual_state = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        actual_action = np.stack(frame["action"].to_numpy()).astype(np.float32)
        expected_shape = (int(item["frames"]), JOINT_DIM)
        _require(
            actual_state.shape == expected_state.shape == expected_shape,
            "State shape mismatch",
        )
        _require(
            actual_action.shape == expected_action.shape == expected_shape,
            "Action shape mismatch",
        )
        _require(np.array_equal(actual_state, expected_state), f"State mismatch: {source}")
        _require(np.array_equal(actual_action, expected_action), f"Action mismatch: {source}")
        _require(np.isfinite(actual_state).all(), f"Non-finite state: {source}")
        _require(np.isfinite(actual_action).all(), f"Non-finite action: {source}")
        _require(instruction == item["instruction"], f"Instruction mismatch: {source}")
        _require(source_fps == FPS, f"Source FPS mismatch: {source}")
        length = len(frame)
        _require(
            np.array_equal(frame["episode_index"].to_numpy(), np.full(length, episode_index)),
            f"episode_index mismatch: {episode_index}",
        )
        _require(
            np.array_equal(frame["frame_index"].to_numpy(), np.arange(length)),
            f"frame_index mismatch: {episode_index}",
        )
        expected_task_index = task_indices[item["instruction"]]
        _require(
            np.array_equal(
                frame["task_index"].to_numpy(),
                np.full(length, expected_task_index),
            ),
            f"task_index mismatch: {episode_index}",
        )
        _require(
            np.array_equal(
                frame["index"].to_numpy(),
                np.arange(next_global_index, next_global_index + length),
            ),
            f"global index mismatch: {episode_index}",
        )
        expected_timestamp = np.arange(length, dtype=np.float32) / np.float32(FPS)
        _require(
            np.allclose(
                frame["timestamp"].to_numpy(dtype=np.float32),
                expected_timestamp,
                rtol=0,
                atol=1e-6,
            ),
            f"timestamp mismatch: {episode_index}",
        )
        next_global_index += length
        if (episode_index + 1) % 50 == 0 or episode_index + 1 == len(manifest):
            print(f"PARQUET_PROGRESS {episode_index + 1}/{len(manifest)}", flush=True)

    _require(next_global_index == context["frame_count"], "Global frame count mismatch")
    print(
        f"PARQUET_EXACT_ALL_OK episodes={len(manifest)} frames={next_global_index} dim={JOINT_DIM}",
        flush=True,
    )


def _check_stats(dataset: Path) -> None:
    stats = json.loads((dataset / "meta/stats.json").read_text(encoding="utf-8"))
    relative = json.loads(
        (dataset / "meta/relative_stats.json").read_text(encoding="utf-8")
    )
    for feature in ("observation.state", "action"):
        for stat in STAT_TYPES:
            value = np.asarray(stats[feature][stat])
            _require(value.shape == (JOINT_DIM,), f"{feature}/{stat} shape mismatch")
            _require(np.isfinite(value).all(), f"{feature}/{stat} is non-finite")
    _require(set(relative) == {"left_arm", "right_arm"}, "Relative stats include non-arm groups")
    for arm in relative:
        for stat in STAT_TYPES:
            value = np.asarray(relative[arm][stat])
            _require(value.shape == (16, 7), f"{arm}/{stat} relative shape mismatch")
            _require(np.isfinite(value).all(), f"{arm}/{stat} is non-finite")
    print("STATS_OK absolute=54 relative_arms=2x16x7 hands=absolute", flush=True)


def _probe_video(path: Path) -> tuple[Path, dict[str, Any]]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,"
        "nb_frames,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    streams = json.loads(completed.stdout)["streams"]
    _require(len(streams) == 1, f"Expected one video stream: {path}")
    return path, streams[0]


def _frame_count(stream: dict[str, Any]) -> int:
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        if value not in (None, "N/A"):
            return int(value)
    return 0


def _check_videos(dataset: Path, context: dict[str, Any], workers: int) -> None:
    _require(shutil.which("ffprobe") is not None, "ffprobe is required")
    manifest = context["manifest"]
    files = sorted(dataset.glob("videos/chunk-*/*/episode_*.mp4"))
    expected_count = len(manifest) * len(VIDEO_MAP)
    _require(len(files) == expected_count, f"Expected {expected_count} videos, got {len(files)}")
    expected_features = {value[0] for value in VIDEO_MAP.values()}
    seen: set[tuple[int, str]] = set()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for path, stream in executor.map(_probe_video, files):
            episode_index = int(path.stem.rsplit("_", 1)[1])
            feature = path.parent.name
            key = (episode_index, feature)
            _require(key not in seen, f"Duplicate video: {path}")
            seen.add(key)
            _require(feature in expected_features, f"Unexpected video feature: {feature}")
            _require(0 <= episode_index < len(manifest), f"Unexpected video episode: {path}")
            _require(
                _frame_count(stream) == int(manifest[episode_index]["frames"]),
                f"Video frame count mismatch: {path}",
            )
            _require(int(stream["width"]) == WIDTH, f"Video width mismatch: {path}")
            _require(int(stream["height"]) == HEIGHT, f"Video height mismatch: {path}")
            _require(Fraction(stream["r_frame_rate"]) == FPS, f"Video FPS mismatch: {path}")
            _require(Fraction(stream["avg_frame_rate"]) == FPS, f"Average FPS mismatch: {path}")
            _require(stream["codec_name"] == VIDEO_CODEC, f"Video codec mismatch: {path}")
            _require(stream["pix_fmt"] == "yuv420p", f"Video pixel format mismatch: {path}")
            _require(path.stat().st_size > 0, f"Empty video: {path}")

    _require(len(seen) == expected_count, "Video coverage mismatch")
    print(
        f"FFPROBE_ALL_OK files={expected_count} fps={FPS} "
        f"resolution={WIDTH}x{HEIGHT} codec={VIDEO_CODEC}",
        flush=True,
    )


def _load_modality_config(config_path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("spark0_joint54_audit_config", config_path)
    _require(spec is not None and spec.loader is not None, f"Cannot load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.tianji_marvin_wuji_config


def _sample_episode_indices(manifest: list[dict[str, Any]]) -> list[int]:
    result: list[int] = []
    for task in TASKS:
        indices = [index for index, item in enumerate(manifest) if item["task"] == task]
        _require(indices, f"No manifest episodes for {task}")
        result.append(indices[0])
        if indices[-1] != indices[0]:
            result.append(indices[-1])
    return result


def _check_official_loader(
    dataset: Path,
    context: dict[str, Any],
    config_path: Path,
    repo_root: Path,
) -> list[int]:
    import gr00t
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

    groot_root = (repo_root / "policy/GR00T_N17/gr00t_n17").resolve(strict=True)
    actual_gr00t = Path(gr00t.__file__).resolve()
    process_data_path = Path(process_data.__file__).resolve()
    _require(
        actual_gr00t.is_relative_to(groot_root),
        f"gr00t resolved outside current checkout: {actual_gr00t}",
    )
    _require(
        process_data_path.is_relative_to(repo_root),
        f"XPolicyLab resolved outside current checkout: {process_data_path}",
    )
    _require(config_path.resolve().is_relative_to(repo_root), "Config escapes current checkout")

    config = _load_modality_config(config_path)
    _require(tuple(config["state"].modality_keys) == PROCESSOR_ORDER, "State order mismatch")
    _require(tuple(config["action"].modality_keys) == PROCESSOR_ORDER, "Action order mismatch")
    representations = [item.rep.name for item in config["action"].action_configs]
    _require(
        representations == ["RELATIVE", "RELATIVE", "ABSOLUTE", "ABSOLUTE"],
        f"Action representations mismatch: {representations}",
    )

    loader = LeRobotEpisodeLoader(dataset, config, video_backend="pyav")
    _require(len(loader) == context["episode_count"], "Official loader length mismatch")
    nested_stats = loader.get_dataset_statistics()
    _require(tuple(nested_stats["state"]) == PROCESSOR_ORDER, "Loader state stats order mismatch")
    _require(tuple(nested_stats["action"]) == PROCESSOR_ORDER, "Loader action stats order mismatch")
    _require(
        set(nested_stats["relative_action"]) == {"left_arm", "right_arm"},
        "Loader relative stats mismatch",
    )

    manifest = context["manifest"]
    sample_indices = _sample_episode_indices(manifest)
    expected_columns = {
        "language.annotation.human.task_description",
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "video.front",
        "video.left_wrist",
        "video.right_wrist",
    }
    for episode_index in sample_indices:
        item = manifest[episode_index]
        source = Path(item["source"])
        expected_state, expected_action, _, _ = _raw_vectors(source)
        frame = loader[episode_index]
        _require(len(frame) == int(item["frames"]), f"Loader length mismatch: {episode_index}")
        _require(
            set(frame.columns) == expected_columns,
            f"Loader columns mismatch: {episode_index}",
        )
        for group in PROCESSOR_ORDER:
            start, end = EXPECTED_SLICES[group]
            for frame_index in (0, len(frame) - 1):
                actual_state = np.asarray(
                    frame[f"state.{group}"].iloc[frame_index], dtype=np.float32
                )
                actual_action = np.asarray(
                    frame[f"action.{group}"].iloc[frame_index], dtype=np.float32
                )
                _require(
                    np.array_equal(actual_state, expected_state[frame_index, start:end]),
                    f"Loader state mismatch: episode={episode_index} group={group}",
                )
                _require(
                    np.array_equal(actual_action, expected_action[frame_index, start:end]),
                    f"Loader action mismatch: episode={episode_index} group={group}",
                )
        language = set(frame["language.annotation.human.task_description"])
        _require(language == {item["instruction"]}, f"Loader language mismatch: {episode_index}")

        image_mae: list[float] = []
        frame_indices = sorted({0, len(frame) // 2, len(frame) - 1})
        for logical_key, (_, hdf5_key) in VIDEO_MAP.items():
            for frame_index in frame_indices:
                decoded = np.asarray(frame[f"video.{logical_key}"].iloc[frame_index])
                source_frame = _decode_hdf5_frame(source, hdf5_key, frame_index)
                _require(
                    decoded.shape == source_frame.shape == (HEIGHT, WIDTH, 3),
                    f"Loader image shape mismatch: episode={episode_index} camera={logical_key}",
                )
                same = float(
                    np.abs(decoded.astype(np.int16) - source_frame.astype(np.int16)).mean()
                )
                swapped = float(
                    np.abs(
                        decoded.astype(np.int16) - source_frame[..., ::-1].astype(np.int16)
                    ).mean()
                )
                _require(
                    same < swapped and same < 20.0,
                    f"Loader RGB mismatch: episode={episode_index} camera={logical_key} "
                    f"frame={frame_index} mae={same:.3f} swapped={swapped:.3f}",
                )
                image_mae.append(same)
        print(
            f"OFFICIAL_LOADER_OK episode={episode_index} task={item['task']} "
            f"frames={len(frame)} image_mae_max={max(image_mae):.3f}",
            flush=True,
        )
        del frame
        gc.collect()

    print(
        f"OFFICIAL_GR00T_SAMPLES_OK samples={len(sample_indices)} "
        f"source={actual_gr00t} backend=pyav",
        flush=True,
    )

    smoke_index = sample_indices[0]
    torchcodec_loader = LeRobotEpisodeLoader(dataset, config, video_backend="torchcodec")
    smoke_frame = torchcodec_loader[smoke_index]
    _require(
        len(smoke_frame) == int(manifest[smoke_index]["frames"]),
        "TorchCodec loader smoke length mismatch",
    )
    smoke_image = np.asarray(smoke_frame["video.front"].iloc[0])
    _require(
        smoke_image.shape == (HEIGHT, WIDTH, 3),
        f"TorchCodec loader smoke image mismatch: {smoke_image.shape}",
    )
    print(
        f"OFFICIAL_GR00T_TORCHCODEC_SMOKE_OK episode={smoke_index}",
        flush=True,
    )
    return sample_indices


def _write_marker(
    dataset: Path,
    context: dict[str, Any],
    config_path: Path,
    source_root: Path,
    sample_indices: list[int],
) -> None:
    marker_path = dataset / MARKER_RELATIVE_PATH
    marker = {
        "version": MARKER_VERSION,
        "status": "passed",
        "dataset_name": dataset.name,
        "robot_type": "tianji_marvin_wuji",
        "episodes": context["episode_count"],
        "frames": context["frame_count"],
        "tasks": len(TASKS),
        "videos": context["episode_count"] * len(VIDEO_MAP),
        "fps": FPS,
        "joint_dim": JOINT_DIM,
        "video_codec": VIDEO_CODEC,
        "source_root": str(source_root),
        "source_manifest_sha256": _sha256_file(context["manifest_path"]),
        "modality_config_sha256": _sha256_file(config_path),
        "audit_script_sha256": _sha256_file(Path(__file__).resolve()),
        "dataset_identity": _dataset_identity(dataset),
        "checks": {
            "parquet_episodes_exact": context["episode_count"],
            "ffprobe_videos": context["episode_count"] * len(VIDEO_MAP),
            "video_codec": VIDEO_CODEC,
            "stats": "absolute54+relative-arms-2x16x7",
            "official_loader": {
                "pyav_episode_indices": sample_indices,
                "torchcodec_smoke_episode_index": sample_indices[0],
            },
        },
    }
    temporary = marker_path.with_name(f"{marker_path.name}.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker_path)
    print(
        f"AUDIT_MARKER_WRITTEN path={marker_path} "
        f"identity={marker['dataset_identity']['sha256']}",
        flush=True,
    )


def _verify_marker(dataset: Path, config_path: Path | None, require_full: bool) -> None:
    marker_path = dataset / MARKER_RELATIVE_PATH
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    _require(marker["version"] == MARKER_VERSION, "Unsupported audit marker version")
    _require(marker["status"] == "passed", "Dataset audit marker is not passed")
    _require(marker["dataset_name"] == dataset.name, "Dataset identity/name mismatch")
    _require(marker["robot_type"] == "tianji_marvin_wuji", "Audit robot mismatch")
    _require(marker["fps"] == FPS and marker["joint_dim"] == JOINT_DIM, "Audit schema mismatch")
    _require(marker["video_codec"] == VIDEO_CODEC, "Audit video codec mismatch")
    _require(marker["tasks"] == len(TASKS), "Audit task count mismatch")
    _require(
        marker["audit_script_sha256"] == _sha256_file(Path(__file__).resolve()),
        "Audit script changed after dataset validation; run the full audit again",
    )
    manifest_path = dataset / "meta/spark0_source_manifest.json"
    _require(
        marker["source_manifest_sha256"] == _sha256_file(manifest_path),
        "Source manifest changed after audit",
    )
    if config_path is not None:
        _require(
            marker["modality_config_sha256"] == _sha256_file(config_path),
            "Modality config changed after audit",
        )
    current_identity = _dataset_identity(dataset)
    _require(
        marker["dataset_identity"] == current_identity,
        "Dataset tree identity changed after audit",
    )

    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    _require(info["total_episodes"] == marker["episodes"], "Marker episode count mismatch")
    _require(info["total_frames"] == marker["frames"], "Marker frame count mismatch")
    _require(info["total_tasks"] == marker["tasks"], "Marker task count mismatch")
    _require(info["total_videos"] == marker["videos"], "Marker video count mismatch")
    _require((dataset / "meta/stats.json").is_file(), "Absolute stats are missing")
    _require((dataset / "meta/relative_stats.json").is_file(), "Relative stats are missing")
    if require_full:
        _require(marker["episodes"] == FULL_EPISODES, "Training requires 600 audited episodes")
        _require(marker["frames"] == FULL_FRAMES, "Training requires 151410 audited frames")
        _require(
            marker["videos"] == FULL_EPISODES * len(VIDEO_MAP),
            "Training video count mismatch",
        )
        _require(
            len(marker["checks"]["official_loader"]["pyav_episode_indices"]) == 12,
            "Full audit must load the first and last episode of every task",
        )
        _require(
            isinstance(
                marker["checks"]["official_loader"]["torchcodec_smoke_episode_index"],
                int,
            ),
            "Full audit is missing the TorchCodec smoke result",
        )
    print(
        f"AUDIT_MARKER_OK episodes={marker['episodes']} frames={marker['frames']} "
        f"identity={current_identity['sha256']}",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--ffprobe-workers", type=int, default=8)
    parser.add_argument("--verify-marker-only", action="store_true")
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = args.dataset.resolve(strict=True)
    _require(dataset.is_dir(), f"Dataset is not a directory: {dataset}")
    config_path = args.config_path.resolve(strict=True) if args.config_path else None
    if args.verify_marker_only:
        _verify_marker(dataset, config_path, args.require_full)
        return

    _require(args.source_root is not None, "--source-root is required for a full audit")
    _require(args.config_path is not None, "--config-path is required for a full audit")
    _require(args.repo_root is not None, "--repo-root is required for a full audit")
    _require(args.ffprobe_workers > 0, "--ffprobe-workers must be positive")
    source_root = args.source_root.resolve(strict=True)
    repo_root = args.repo_root.resolve(strict=True)
    context = _check_metadata(
        dataset,
        source_root,
        args.expected_episodes,
        args.require_full,
    )
    _check_parquet(dataset, context)
    _check_stats(dataset)
    _check_videos(dataset, context, args.ffprobe_workers)
    sample_indices = _check_official_loader(dataset, context, config_path, repo_root)
    _write_marker(dataset, context, config_path, source_root, sample_indices)
    _verify_marker(dataset, config_path, args.require_full)
    print("SPARK0_GROOT_V21_AUDIT_ALL_OK", flush=True)


if __name__ == "__main__":
    main()
