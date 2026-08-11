#!/usr/bin/env python3
"""Convert the six raw Spark0 tasks to a fresh GR00T-flavored LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import lerobot.datasets.lerobot_dataset as lerobot_dataset_module
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.datasets.video_utils import encode_video_frames
from XPolicyLab.utils import process_data


TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
CAMERAS = {
    "cam_high": ("vision", "cam_head", "colors"),
    "cam_left_wrist": ("vision", "cam_left_wrist", "colors"),
    "cam_right_wrist": ("vision", "cam_right_wrist", "colors"),
}
HEIGHT = 480
WIDTH = 640
FPS = 25
JOINT_DIM = 54
VIDEO_CODEC = "h264"
VIDEO_ENCODING_WORKERS = 3


def _encode_episode_videos_h264(dataset: LeRobotDataset, episode_index: int) -> None:
    """Encode the three independent camera streams concurrently as H.264."""

    jobs: list[tuple[Path, Path]] = []
    for key in dataset.meta.video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        if video_path.is_file():
            continue
        image_dir = dataset._get_image_file_path(
            episode_index=episode_index,
            image_key=key,
            frame_index=0,
        ).parent
        jobs.append((image_dir, video_path))

    def encode(job: tuple[Path, Path]) -> None:
        image_dir, video_path = job
        encode_video_frames(
            image_dir,
            video_path,
            dataset.fps,
            vcodec=VIDEO_CODEC,
            log_level=None,
            overwrite=True,
        )

    if jobs:
        with ThreadPoolExecutor(max_workers=min(VIDEO_ENCODING_WORKERS, len(jobs))) as executor:
            list(executor.map(encode, jobs))
        for image_dir, _ in jobs:
            shutil.rmtree(image_dir)

    if dataset.meta.video_keys and episode_index == 0:
        dataset.meta.update_video_info()
        lerobot_dataset_module.write_info(dataset.meta.info, dataset.meta.root)


def _configure_video_encoding() -> None:
    """Use H.264 for LeRobot v2.1 episode files.

    LeRobot 0.3.3's ``video_backend`` argument selects the decoder; its writer
    otherwise calls ``encode_video_frames`` with the libsvtav1 default. Replace
    the encoder method only in this conversion process, and encode the three
    independent camera streams concurrently. The installed package and GR00T
    data-loading code remain unchanged.
    """

    LeRobotDataset.encode_episode_videos = _encode_episode_videos_h264


def _add_frame_compat(dataset: LeRobotDataset, frame: dict[str, Any], task: str) -> None:
    try:
        dataset.add_frame(frame, task=task)
    except TypeError:
        dataset.add_frame({**frame, "task": task})


def _save_episode_compat(dataset: LeRobotDataset, task: str) -> None:
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def _release_dataset_memory(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "create_hf_dataset") and hasattr(dataset, "hf_dataset"):
        try:
            dataset.hf_dataset = dataset.create_hf_dataset()
        except Exception:
            # Older LeRobot builds do not expose a stable public release hook.
            pass


def _decode_instruction(value: Any) -> str:
    values = np.asarray(value).reshape(-1)
    for item in values:
        if isinstance(item, np.generic):
            item = item.item()
        if isinstance(item, (bytes, bytearray, memoryview)):
            text = bytes(item).rstrip(b"\0").decode("utf-8")
        else:
            text = str(item)
        if text.strip():
            return text.strip()
    return ""


def _episode_files(source_root: Path, task: str) -> list[Path]:
    data_root = source_root / task / "tianji_marvin_wuji/data"
    files = sorted(data_root.glob("episode_*.hdf5"))
    files.extend(sorted(data_root.glob("episode_*.h5")))
    unique = list(dict.fromkeys(path.resolve() for path in files))
    if len(unique) != 100:
        raise RuntimeError(f"{task}: expected exactly 100 raw episodes, got {len(unique)}")
    return unique


def _create_dataset(repo_id: str, image_writer_threads: int) -> LeRobotDataset:
    names = (
        [f"left_arm_joint_{index}" for index in range(7)]
        + [f"left_hand_joint_{index}" for index in range(20)]
        + [f"right_arm_joint_{index}" for index in range(7)]
        + [f"right_hand_joint_{index}" for index in range(20)]
    )
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [names],
        },
        "action": {
            "dtype": "float32",
            "shape": (JOINT_DIM,),
            "names": [names],
        },
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, HEIGHT, WIDTH),
            "names": ["channels", "height", "width"],
        }
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="tianji_marvin_wuji",
        features=features,
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
    )


def _raw_vector(handle: h5py.File, prefix: str) -> np.ndarray:
    return np.concatenate(
        [
            handle[f"{prefix}/left_arm_joint_states"][:],
            handle[f"{prefix}/left_ee_joint_states"][:],
            handle[f"{prefix}/right_arm_joint_states"][:],
            handle[f"{prefix}/right_ee_joint_states"][:],
        ],
        axis=1,
    ).astype(np.float32)


def _validate_source_root(source_root: Path) -> dict[str, list[Path]]:
    discovered = sorted(
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and (path / "tianji_marvin_wuji/data").is_dir()
    )
    if discovered != sorted(TASKS):
        raise RuntimeError(f"Expected exactly the six Spark0 raw tasks, got {discovered}")
    return {task: _episode_files(source_root, task) for task in TASKS}


def _convert_episode(
    path: Path,
    task: str,
    dataset: LeRobotDataset,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        source_fps = int(handle["additional_info/frequency"][()])
        if source_fps != FPS:
            raise ValueError(f"{path}: expected {FPS} Hz, got {source_fps}")
        state = _raw_vector(handle, "state")
        action = _raw_vector(handle, "action")
        if state.ndim != 2 or state.shape[1] != JOINT_DIM or action.shape != state.shape:
            raise ValueError(f"{path}: invalid state/action shapes {state.shape}/{action.shape}")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError(f"{path}: state/action contains non-finite values")

        instruction = _decode_instruction(handle["instruction"][()])
        if not instruction:
            raise ValueError(f"{path}: empty instruction")

        images: dict[str, np.ndarray] = {}
        for camera, keys in CAMERAS.items():
            shape_key = "/".join(keys[:-1] + ("shape",))
            source_shape = tuple(int(value) for value in handle[shape_key][()])
            if source_shape != (HEIGHT, WIDTH, 3):
                raise ValueError(f"{path}/{camera}: source shape is {source_shape}")
            frames = np.asarray(process_data.decode_image_bit(handle["/".join(keys)][:]))
            expected_shape = (len(state), HEIGHT, WIDTH, 3)
            if frames.shape != expected_shape or frames.dtype != np.uint8:
                raise ValueError(
                    f"{path}/{camera}: decoded image contract mismatch: "
                    f"{frames.shape}/{frames.dtype}"
                )
            images[camera] = frames

    for frame_index in range(len(state)):
        frame = {
            "observation.state": state[frame_index],
            "action": action[frame_index],
        }
        for camera, frames in images.items():
            frame[f"observation.images.{camera}"] = frames[frame_index]
        _add_frame_compat(dataset, frame, instruction)
    _save_episode_compat(dataset, instruction)
    _release_dataset_memory(dataset)
    print(f"EPISODE_OK task={task} file={path.name} frames={len(state)}", flush=True)
    return {
        "task": task,
        "source": str(path),
        "frames": len(state),
        "instruction": instruction,
        "fps": source_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    args = parser.parse_args()

    if importlib.metadata.version("lerobot") != "0.3.3":
        raise RuntimeError("Spark0 v2.1 conversion requires lerobot==0.3.3")
    if not 1 <= args.episodes_per_task <= 100:
        raise ValueError("--episodes-per-task must be between 1 and 100")
    if args.image_writer_threads <= 0:
        raise ValueError("--image-writer-threads must be positive")

    source_root = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    repo_root = args.repo_root.resolve(strict=True)
    process_data_path = Path(process_data.__file__).resolve()
    if not process_data_path.is_relative_to(repo_root):
        raise RuntimeError(
            f"XPolicyLab process_data resolved outside current repository: {process_data_path}"
        )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if output.parent != Path(HF_LEROBOT_HOME).resolve():
        raise RuntimeError(
            f"--output parent must equal HF_LEROBOT_HOME: {output.parent} != {HF_LEROBOT_HOME}"
        )
    if output.name.startswith(".") or "/" in output.name:
        raise ValueError(f"Unsafe output dataset name: {output.name!r}")

    _configure_video_encoding()
    print(f"Video codec: {VIDEO_CODEC}", flush=True)
    episode_files = _validate_source_root(source_root)
    dataset = _create_dataset(output.name, args.image_writer_threads)
    manifest = []
    for task in TASKS:
        for path in episode_files[task][: args.episodes_per_task]:
            manifest.append(_convert_episode(path, task, dataset))

    meta = output / "meta"
    (meta / "spark0_source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"CONVERSION_OK output={output} episodes={len(manifest)} "
        f"frames={sum(item['frames'] for item in manifest)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
