#!/usr/bin/env python3
"""Read-only compatibility audit for Spark0 joint54 and XPL GR00T N1.7."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py


DEFAULT_SOURCE = Path("/mnt/xspark-data/tjy/spark0_bench")
EXPECTED_MAX_DIM = 132
EXPECTED_ACTION_HORIZON = 40
EXPECTED_TOTAL_EPISODES = 600
EXPECTED_TOTAL_FRAMES = 151410
EXPECTED_TASKS = (
    "collect_objects",
    "dual_bottles_pick",
    "hammer_beat",
    "insert_block",
    "retrieve_gap",
    "stack_bowls",
)
EXPECTED_GROUP_DIMS = (
    ("left_arm_joint_states", 7),
    ("left_ee_joint_states", 20),
    ("right_arm_joint_states", 7),
    ("right_ee_joint_states", 20),
)
EXPECTED_MODALITY_RANGES = {
    "left_arm": (0, 7),
    "left_hand": (7, 27),
    "right_arm": (27, 34),
    "right_hand": (34, 54),
}

EXIT_SOURCE_CONTRACT_MISMATCH = 3
EXIT_MODEL_ENVELOPE_MISMATCH = 4
EXIT_ADAPTER_REVIEW_REQUIRED = 5


def _read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def _parse_int_assignment(text: str, name: str) -> int:
    match = re.search(rf"^\s*{re.escape(name)}\s*:\s*int\s*=\s*(\d+)", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"could not find integer assignment for {name}")
    return int(match.group(1))


def _group_dims(handle: h5py.File, prefix: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for name, _ in EXPECTED_GROUP_DIMS:
        key = f"{prefix}/{name}"
        if key not in handle:
            raise KeyError(f"missing source dataset: {key}")
        dataset = handle[key]
        if dataset.ndim != 2:
            raise ValueError(f"{key} must be 2D, got shape {dataset.shape}")
        result[name] = int(dataset.shape[-1])
    return result


def _waist_paths(handle: h5py.File) -> list[str]:
    paths: list[str] = []
    for prefix in ("state", "action"):
        if prefix not in handle:
            continue

        def visit(name: str, _: Any, *, root: str = prefix) -> None:
            if "waist" in name.lower():
                paths.append(f"{root}/{name}")

        handle[prefix].visititems(visit)
    return sorted(paths)


def _scalar_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _inspect_source(source_root: Path) -> dict[str, Any]:
    expected_dims = dict(EXPECTED_GROUP_DIMS)
    task_reports: dict[str, Any] = {}
    all_valid = True

    discovered_tasks = sorted(
        path.name
        for path in source_root.iterdir()
        if path.is_dir() and (path / "tianji_marvin_wuji/data").is_dir()
    )
    if discovered_tasks != sorted(EXPECTED_TASKS):
        raise ValueError(
            f"expected exactly six raw tasks {sorted(EXPECTED_TASKS)}, got {discovered_tasks}"
        )

    for task in EXPECTED_TASKS:
        episodes = sorted(
            (source_root / task / "tianji_marvin_wuji/data").glob("episode_*.hdf5")
        )
        if not episodes:
            raise FileNotFoundError(f"no source episodes for task {task}")

        frequencies: set[int] = set()
        frame_count = 0
        task_valid = True
        for episode in episodes:
            with h5py.File(episode, "r") as handle:
                state_groups = _group_dims(handle, "state")
                action_groups = _group_dims(handle, "action")
                if state_groups != expected_dims or action_groups != expected_dims:
                    task_valid = False

                horizons = {
                    int(handle[f"{prefix}/{name}"].shape[0])
                    for prefix in ("state", "action")
                    for name, _ in EXPECTED_GROUP_DIMS
                }
                if len(horizons) != 1:
                    raise ValueError(f"state/action horizon mismatch in {episode}: {horizons}")
                horizon = horizons.pop()
                frame_count += horizon
                if _waist_paths(handle):
                    task_valid = False
                frequencies.add(int(handle["additional_info/frequency"][()]))
                if not _scalar_text(handle["instruction"][()]).strip():
                    raise ValueError(f"empty instruction in {episode}")
                for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                    colors = handle[f"vision/{camera}/colors"]
                    if int(colors.shape[0]) != horizon:
                        raise ValueError(f"camera horizon mismatch in {episode}: {camera}")
                    shape = tuple(int(value) for value in handle[f"vision/{camera}/shape"][()])
                    if shape != (480, 640, 3):
                        raise ValueError(f"camera shape mismatch in {episode}: {camera}={shape}")

        task_valid = task_valid and frequencies == {25} and len(episodes) == 100
        all_valid = all_valid and task_valid
        task_reports[task] = {
            "episodes": len(episodes),
            "frames": frame_count,
            "frequencies": sorted(frequencies),
            "matches_joint54": task_valid,
        }

    total_episodes = sum(item["episodes"] for item in task_reports.values())
    total_frames = sum(item["frames"] for item in task_reports.values())
    return {
        "tasks": task_reports,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "state_dim": 54,
        "action_dim": 54,
        "fps": 25,
        "no_waist": True,
        "matches_joint54": (
            all_valid
            and total_episodes == EXPECTED_TOTAL_EPISODES
            and total_frames == EXPECTED_TOTAL_FRAMES
        ),
    }


def _inspect_modality_config(policy_root: Path) -> dict[str, Any]:
    config_path = policy_root / "configs/tianji_marvin_wuji_config.py"
    modality_path = policy_root / "configs/tianji_marvin_wuji_modality.json"
    config_text = _read(config_path)
    modality = json.loads(_read(modality_path))

    actual_ranges = {
        section: {
            key: (int(value["start"]), int(value["end"]))
            for key, value in modality[section].items()
        }
        for section in ("state", "action")
    }
    group_literal = '["left_arm", "right_arm", "left_hand", "right_hand"]'
    return {
        "state_ranges": actual_ranges["state"],
        "action_ranges": actual_ranges["action"],
        "python_group_order_matches": config_text.count(group_literal) >= 2,
        "matches_joint54": (
            actual_ranges["state"] == EXPECTED_MODALITY_RANGES
            and actual_ranges["action"] == EXPECTED_MODALITY_RANGES
            and config_text.count(group_literal) >= 2
            and "ActionRepresentation.RELATIVE" in config_text
            and config_text.count("ActionRepresentation.ABSOLUTE") == 2
            and "list(range(0, 16))" in config_text
        ),
    }


def _inspect_local_code(repo_root: Path) -> dict[str, Any]:
    policy_root = repo_root / "policy/GR00T_N17"
    upstream = policy_root / "gr00t_n17/gr00t"
    model_text = _read(policy_root / "model.py")
    process_text = _read(policy_root / "process_data.sh")
    converter_shell_text = _read(
        policy_root / "scripts/convert_spark0_hdf5_to_groot_v21.sh"
    )
    converter_python_text = _read(
        policy_root / "scripts/convert_spark0_hdf5_to_groot_v21.py"
    )
    converter_text = converter_shell_text + converter_python_text
    audit_text = _read(policy_root / "scripts/audit_spark0_groot_v21.py")
    train_recipe_text = _read(policy_root / "scripts/train_spark0_groot_joint54.sh")
    config_text = _read(upstream / "configs/model/gr00t_n1d7.py")
    processor_text = _read(upstream / "model/gr00t_n1d7/processing_gr00t_n1d7.py")
    loss_text = _read(upstream / "model/gr00t_n1d7/gr00t_n1d7.py")

    max_state_dim = _parse_int_assignment(config_text, "max_state_dim")
    max_action_dim = _parse_int_assignment(config_text, "max_action_dim")
    action_horizon = _parse_int_assignment(config_text, "action_horizon")
    state_padding = "self.max_state_dim - normalized_states.shape[1]" in processor_text
    action_padding = "self.max_action_dim - normalized_actions.shape[1]" in processor_text
    horizon_mask = "action_mask[action_horizon:] = 0" in processor_text
    dimension_mask = "action_mask[:, action_dim:] = 0" in processor_text
    masked_loss = (
        re.search(r"mse_loss\([^\n]+reduction=[\"']none[\"']\)\s*\*\s*action_mask", loss_text)
        is not None
        and "action_mask.sum()" in loss_text
    )
    exact_envelope = (
        max_state_dim == EXPECTED_MAX_DIM
        and max_action_dim == EXPECTED_MAX_DIM
        and action_horizon == EXPECTED_ACTION_HORIZON
    )
    masked_padding = all(
        (state_padding, action_padding, horizon_mask, dimension_mask, masked_loss)
    )

    model_metadata_driven = all(
        token in model_text
        for token in (
            "get_robot_action_dim_info",
            "_resolve_robot_action_dim_info",
            "except FileNotFoundError",
            "_robot_info.json",
            "pack_robot_state",
            "unpack_robot_state",
            'f"{side}_hand"',
            "expected_groups",
        )
    )
    old_hardcoded_contract_absent = not any(
        token in model_text for token in ("left[:6]", "left[6:7]", "right[:6]", "right[6:7]")
    )
    process_dispatches_spark = all(
        token in process_text
        for token in ("tianji_marvin_wuji", "convert_spark0_hdf5_to_groot_v21.sh")
    )
    conversion_guarded = all(
        token in converter_text
        for token in (
            "GR00T_SPARK0_HDF5_ROOT",
            "sim_6tasks_lerobot",
            "refusing reconversion",
            "fps=25",
            "HEIGHT = 480",
            "WIDTH = 640",
            "FPS = 25",
            "cam_right_wrist",
            "process_data.decode_image_bit",
        )
    ) and "transform_lerobot_v21_format.py" not in converter_python_text
    conversion_independent_audit = all(
        token in converter_shell_text
        for token in (
            "audit_spark0_groot_v21.py",
            "--expected-episodes",
            "--ffprobe-workers",
            "lock_acquired=0",
            "lock_acquired=1",
            "mv -T",
        )
    ) and all(
        token in audit_text
        for token in (
            "PARQUET_EXACT_ALL_OK",
            "FFPROBE_ALL_OK",
            "OFFICIAL_GR00T_SAMPLES_OK",
            "spark0_joint54_audit.json",
            "process_data.decode_image_bit",
            "_dataset_identity",
        )
    )
    recipe_exact = all(
        token in train_recipe_text
        for token in (
            "export NUM_GPUS=8",
            "export GLOBAL_BATCH_SIZE=64",
            "export MAX_STEPS=100000",
            "export SAVE_STEPS=5000",
            "export USE_WANDB=1",
            "export WANDB_MODE=online",
            "refusing implicit resume",
            "groot_n17_lerobotV21_joint54",
            "pretrain_model/GR00T-N1.7-3B",
            "pretrain_model/Cosmos-Reason2-2B",
            "HF_HUB_OFFLINE=1",
            "--verify-marker-only",
            "--require-full",
        )
    )

    modality = _inspect_modality_config(policy_root)
    return {
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "action_horizon": action_horizon,
        "masked_padding_verified": masked_padding,
        "base_model_supports_joint54_without_head_resize": exact_envelope and masked_padding,
        "model_is_metadata_driven": model_metadata_driven,
        "old_6_plus_1_output_slices_absent": old_hardcoded_contract_absent,
        "process_dispatches_spark0": process_dispatches_spark,
        "conversion_is_raw_guarded_25hz_480x640": conversion_guarded,
        "conversion_has_pre_publish_independent_audit": conversion_independent_audit,
        "training_recipe_is_exact": recipe_exact,
        "modality": modality,
        "adapter_supports_joint54": (
            model_metadata_driven
            and old_hardcoded_contract_absent
            and process_dispatches_spark
            and conversion_guarded
            and conversion_independent_audit
            and modality["matches_joint54"]
        ),
    }


def _emit(report: dict[str, Any], classification: str, exit_code: int) -> int:
    report["classification"] = classification
    report["exit_code"] = exit_code
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Spark0 joint54 against the local XPL GR00T N1.7 integration."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report: dict[str, Any] = {
        "compatible": False,
        "expected_spark0_layout": [
            "left_arm:7",
            "left_hand:20",
            "right_arm:7",
            "right_hand:20",
        ],
        "expected_spark0_dim": 54,
        "head_resize_required": False,
        "fresh_joint54_checkpoint_required": True,
        "checker_is_read_only": True,
    }

    try:
        report["source"] = _inspect_source(args.source_root.resolve())
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        report["source_error"] = f"{type(exc).__name__}: {exc}"
        return _emit(report, "SOURCE_CONTRACT_MISMATCH", EXIT_SOURCE_CONTRACT_MISMATCH)
    if not report["source"]["matches_joint54"]:
        return _emit(report, "SOURCE_CONTRACT_MISMATCH", EXIT_SOURCE_CONTRACT_MISMATCH)

    try:
        report["local_code"] = _inspect_local_code(repo_root)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        report["local_code_error"] = f"{type(exc).__name__}: {exc}"
        return _emit(report, "MODEL_ENVELOPE_MISMATCH", EXIT_MODEL_ENVELOPE_MISMATCH)

    local = report["local_code"]
    if not local["base_model_supports_joint54_without_head_resize"]:
        report["head_resize_required"] = None
        return _emit(report, "MODEL_ENVELOPE_MISMATCH", EXIT_MODEL_ENVELOPE_MISMATCH)
    if not local["adapter_supports_joint54"] or not local["training_recipe_is_exact"]:
        report["reason"] = "The joint54 adapter/config/recipe differs from the reviewed contract."
        return _emit(report, "ADAPTER_REVIEW_REQUIRED", EXIT_ADAPTER_REVIEW_REQUIRED)

    report["compatible"] = True
    report["reason"] = (
        "Spark0 joint54 fits the unchanged N1.7 132D masked envelope. The XPolicyLab "
        "adapter, raw-HDF5 conversion path, modality config, and fresh 8-GPU training "
        "recipe all match the reviewed 7/20/7/20 contract."
    )
    return _emit(report, "COMPATIBLE_FRESH_JOINT54_CHECKPOINT_REQUIRED", 0)


if __name__ == "__main__":
    raise SystemExit(main())
