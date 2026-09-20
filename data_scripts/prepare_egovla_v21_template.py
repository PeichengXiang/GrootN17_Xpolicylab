#!/usr/bin/env python3
"""Convert the audited EgoVLA LeRobot v3 donor into a v2.1 template.

The donor is never renamed or modified.  NVIDIA's vendored v3-to-v2 helpers
split its consolidated parquet/video shards into the episode-oriented layout
required by GR00T N1.7.  XPolicyLab provenance is then preserved under the
name expected by ``convert_egovla_to_groot.py``, modality metadata and fresh
absolute/relative statistics are generated, and the result is published by a
single rename only after the normal raw/cache validator passes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any

import av
import convert_egovla_to_groot as egovla
import h5py
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ROOT = SCRIPT_DIR.parent
DEFAULT_SOURCE_V30 = MODEL_ROOT.parent / "pi05" / "data" / "EgoVLA_benchmark"
DEFAULT_OUTPUT = MODEL_ROOT / "data" / "EgoVLA_benchmark_template_v21"
DEFAULT_RAW = Path("/personal/xiangpc/EgoVLA benchmark/data/EgoVLA/raw_remove_deprecated")
DEFAULT_GR00T_ROOT = MODEL_ROOT / "policy" / "GR00T_N17" / "gr00t_n17"
DEFAULT_MODALITY_CONFIG = (
    MODEL_ROOT / "policy" / "GR00T_N17" / "configs" / "ego_h1_inspire_config.py"
)


def load_upstream_converter(gr00t_root: Path) -> ModuleType:
    try:
        lerobot_version = version("lerobot")
    except PackageNotFoundError as exc:
        raise RuntimeError("prepare requires the pinned lerobot==0.4.4 package") from exc
    if lerobot_version != "0.4.4":
        raise RuntimeError(f"prepare requires lerobot==0.4.4, found {lerobot_version!r}")
    print(f"CONVERTER_DEPENDENCY_OK lerobot={lerobot_version}", flush=True)
    path = gr00t_root / "scripts" / "lerobot_conversion" / "convert_v3_to_v2.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("groot_convert_v3_to_v2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(job: tuple[Path, str, int, bool]) -> dict[str, Any]:
    path, relative_path, expected_frames, expect_black = job
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"{path}: expected one video stream, got {len(streams)}")
    stream = streams[0]
    actual_frames = int(stream.get("nb_read_frames", -1))
    if (
        actual_frames != expected_frames
        or int(stream.get("width", -1)) != 384
        or int(stream.get("height", -1)) != 384
        or stream.get("r_frame_rate") != "30/1"
    ):
        raise ValueError(
            f"{path}: frames/shape/fps={actual_frames}/"
            f"{stream.get('height')}x{stream.get('width')}/"
            f"{stream.get('r_frame_rate')}, expected={expected_frames}/384x384/30"
        )
    black_max = None
    black_frames = 0
    if expect_black:
        black_max = 0
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                array = frame.to_ndarray(format="rgb24")
                black_max = max(black_max, int(array.max(initial=0)))
                black_frames += 1
        if black_frames != expected_frames or black_max != 0:
            raise ValueError(
                f"{path}: expected {expected_frames} exactly black frames, "
                f"got frames={black_frames}, max={black_max}"
            )
    return {
        "relative_path": relative_path,
        "frames": actual_frames,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "black_frames": black_frames,
        "black_max": black_max,
    }


def decode_first_rgb(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        frame = next(container.decode(video=0))
        return frame.to_ndarray(format="rgb24")


def video_path(dataset: Path, key: str, episode_index: int, chunks_size: int) -> Path:
    return (
        dataset
        / "videos"
        / f"chunk-{episode_index // chunks_size:03d}"
        / key
        / f"episode_{episode_index:06d}.mp4"
    )


def audit_v21_videos(
    dataset: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    chunks_size: int,
    workers: int,
    source_manifest: dict[str, Any],
    raw_source: Path,
) -> None:
    provenance = source_manifest.get("episodes")
    if not isinstance(provenance, list) or len(provenance) != 1903:
        raise ValueError("v3 source manifest must contain exactly 1903 episodes")
    provenance_by_index = {int(item["dataset_episode_index"]): item for item in provenance}
    if sorted(provenance_by_index) != list(range(1903)):
        raise ValueError("v3 source manifest episode indices are not contiguous")

    jobs_by_key: dict[str, list[tuple[Path, str, int, bool]]] = {key: [] for key in video_keys}
    for row in sorted(episode_records, key=lambda item: int(item["episode_index"])):
        episode_index = int(row["episode_index"])
        frames = int(row["dataset_to_index"]) - int(row["dataset_from_index"])
        episode_chunk = episode_index // chunks_size
        for key in video_keys:
            relative = (
                Path("videos")
                / f"chunk-{episode_chunk:03d}"
                / key
                / f"episode_{episode_index:06d}.mp4"
            )
            path = dataset / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            expect_black = key != "observation.images.cam_high" and not bool(
                provenance_by_index[episode_index]["has_real_wrist_cameras"]
            )
            jobs_by_key[key].append((path, relative.as_posix(), frames, expect_black))

    summary: dict[str, Any] = {
        "schema": "egovla-v21-video-audit-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workers": workers,
        "cameras": {},
    }
    for key, jobs in jobs_by_key.items():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(probe_video, jobs))
        total_frames = sum(int(row["frames"]) for row in rows)
        if len(rows) != 1903 or total_frames != 510546:
            raise ValueError(
                f"{key}: expected 1903 videos/510546 frames, got {len(rows)}/{total_frames}"
            )
        inventory = hashlib.sha256()
        for row in sorted(rows, key=lambda item: str(item["relative_path"])):
            inventory.update(
                (f"{row['relative_path']}\0{row['size_bytes']}\0{row['sha256']}\n").encode()
            )
        black_frames = sum(int(row["black_frames"]) for row in rows)
        black_max = max(
            (int(row["black_max"]) for row in rows if row["black_max"] is not None),
            default=None,
        )
        if key != "observation.images.cam_high" and (black_frames != 194547 or black_max != 0):
            raise ValueError(
                f"{key}: expected 194547 exactly black frames, got {black_frames}/{black_max}"
            )
        summary["cameras"][key] = {
            "videos": len(rows),
            "frames": total_frames,
            "bytes": sum(int(row["size_bytes"]) for row in rows),
            "inventory_sha256": inventory.hexdigest(),
            "verified_black_frames": black_frames,
            "verified_black_max": black_max,
        }
        print(f"VIDEO_AUDIT_OK key={key} videos={len(rows)} frames={total_frames}")

    first_by_task: dict[str, dict[str, Any]] = {}
    for item in provenance:
        first_by_task.setdefault(str(item["task"]), item)
    color_alignment: dict[str, Any] = {}
    for task, item in first_by_task.items():
        episode_index = int(item["dataset_episode_index"])
        raw_path = raw_source / str(item["source_relative_path"])
        with h5py.File(raw_path, "r") as h5:
            raw_head = np.asarray(h5["observations/images/main"][0], dtype=np.uint8)
            samples = {
                "observation.images.cam_high": raw_head,
            }
            if bool(item["has_real_wrist_cameras"]):
                samples.update(
                    {
                        "observation.images.cam_left_wrist": np.asarray(
                            h5["observations/images/left_hand"][0], dtype=np.uint8
                        ),
                        "observation.images.cam_right_wrist": np.asarray(
                            h5["observations/images/right_hand"][0], dtype=np.uint8
                        ),
                    }
                )
        task_results: dict[str, Any] = {}
        for key, raw_frame in samples.items():
            decoded = decode_first_rgb(video_path(dataset, key, episode_index, chunks_size))
            direct = float(np.abs(decoded.astype(np.int16) - raw_frame.astype(np.int16)).mean())
            swapped = float(
                np.abs(decoded.astype(np.int16) - raw_frame[..., ::-1].astype(np.int16)).mean()
            )
            if not direct < swapped:
                raise ValueError(
                    f"{task}/{key}: RGB alignment failed, direct={direct}, swapped={swapped}"
                )
            task_results[key] = {
                "direct_rgb_mae": direct,
                "channel_swapped_mae": swapped,
            }
        color_alignment[task] = task_results
    if len(color_alignment) != 12:
        raise ValueError(f"expected 12 RGB task samples, got {len(color_alignment)}")
    summary["rgb_alignment"] = color_alignment
    egovla._write_json(dataset / "meta" / "egovla_v21_video_audit.json", summary)


def convert(
    source: Path,
    output: Path,
    raw_source: Path,
    gr00t_root: Path,
    modality_config: Path,
    video_workers: int,
) -> None:
    source = source.resolve()
    output = output.resolve()
    raw_source = raw_source.resolve()
    gr00t_root = gr00t_root.resolve()
    modality_config = modality_config.resolve()
    if video_workers < 1:
        raise ValueError(f"video_workers must be positive, got {video_workers}")
    if shutil.which("ffprobe") is None:
        raise FileNotFoundError("ffprobe is required for the full video audit")
    for required in (
        source / "meta" / "info.json",
        source / "meta" / "xpolicylab_conversion.json",
        modality_config,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    raw_audit = egovla.audit_raw(raw_source)
    if output.exists():
        existing = egovla.validate_cache(output, raw_audit)
        print(
            f"ALREADY_VALID output={output} episodes={existing['total_episodes']}",
            flush=True,
        )
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.staging-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)

    converter = load_upstream_converter(gr00t_root)
    converter.validate_local_dataset_version(source)
    episode_records = converter.load_episode_records(source)
    info = converter.load_info(source)
    source_manifest = egovla._read_json(source / "meta" / "xpolicylab_conversion.json")
    video_keys = [
        key for key, feature in info["features"].items() if feature.get("dtype") == "video"
    ]
    expected_video_keys = [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]
    if video_keys != expected_video_keys:
        raise ValueError(f"unexpected v3 video keys/order: {video_keys}")
    chunks_size = int(info.get("chunks_size", converter.DEFAULT_CHUNK_SIZE))

    try:
        staging.mkdir(parents=True, exist_ok=False)
        converter.convert_info(source, staging, episode_records, video_keys)
        converter.copy_global_stats(source, staging)
        converter.convert_tasks(source, staging)
        converter.convert_data(source, staging, episode_records, chunks_size)
        converter.convert_videos(source, staging, episode_records, video_keys, chunks_size)
        converter.convert_episodes_metadata(staging, episode_records)
        converter.copy_ancillary_directories(source, staging)
        audit_v21_videos(
            staging,
            episode_records,
            video_keys,
            chunks_size,
            video_workers,
            source_manifest,
            raw_source,
        )

        source_manifest = source / "meta" / "xpolicylab_conversion.json"
        shutil.copy2(
            source_manifest,
            staging / "meta" / "xpolicylab_source_conversion.json",
        )
        shutil.copy2(
            source_manifest,
            staging / "meta" / "template_xpolicylab_v30_conversion.json",
        )
        egovla._write_json(staging / "meta" / "modality.json", egovla.GR00T_MODALITY)
        egovla._regenerate_statistics(staging, gr00t_root, modality_config)

        cache_audit = egovla.validate_cache(staging, raw_audit)
        staging.replace(output)
        print(
            f"DONE output={output} episodes={cache_audit['total_episodes']} "
            f"frames={cache_audit['total_frames']} source_unchanged={source}",
            flush=True,
        )
    except Exception:
        print(f"FAILED; partial staging retained at {staging}", file=sys.stderr)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v30", type=Path, default=DEFAULT_SOURCE_V30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--gr00t-root", type=Path, default=DEFAULT_GR00T_ROOT)
    parser.add_argument("--modality-config", type=Path, default=DEFAULT_MODALITY_CONFIG)
    parser.add_argument("--video-workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"PREPARE source_v30={args.source_v30} output={args.output} "
        f"started_at={datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        flush=True,
    )
    convert(
        args.source_v30,
        args.output,
        args.source_raw,
        args.gr00t_root,
        args.modality_config,
        args.video_workers,
    )


if __name__ == "__main__":
    main()
