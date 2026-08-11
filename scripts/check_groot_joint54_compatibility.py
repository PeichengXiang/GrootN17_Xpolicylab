#!/usr/bin/env python3
"""Read-only compatibility audit for Spark0 joint54 and XPL GR00T N1.7.

The expected current result is exit 2: the N1.7 base model has a verified 132D
state/action envelope with masked padding, while the official XPolicyLab
adapter is still the 14D ARX contract.  The checker never converts data, edits
the adapter, or starts training.
"""

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
EXPECTED_GROUP_DIMS = (
    ("left_arm_joint_states", 7),
    ("left_ee_joint_states", 20),
    ("right_arm_joint_states", 7),
    ("right_ee_joint_states", 20),
)

EXIT_ADAPTER_CHANGE_REQUIRED = 2
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
        if dataset.ndim < 2:
            raise ValueError(f"{key} must have a joint dimension, got shape {dataset.shape}")
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


def _inspect_source(source_root: Path) -> dict[str, Any]:
    episodes = sorted(source_root.glob("*/tianji_marvin_wuji/data/episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"no source episodes below {source_root}")

    with h5py.File(episodes[0], "r") as handle:
        state_groups = _group_dims(handle, "state")
        action_groups = _group_dims(handle, "action")
        waist_paths = _waist_paths(handle)

    expected = dict(EXPECTED_GROUP_DIMS)
    state_dim = sum(state_groups.values())
    action_dim = sum(action_groups.values())
    return {
        "episode": str(episodes[0]),
        "state_groups": state_groups,
        "action_groups": action_groups,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "waist_paths": waist_paths,
        "no_waist": not waist_paths,
        "matches_joint54": (
            state_groups == expected
            and action_groups == expected
            and state_dim == 54
            and action_dim == 54
            and not waist_paths
        ),
    }


def _inspect_local_code(repo_root: Path) -> dict[str, Any]:
    policy_root = repo_root / "policy/GR00T_N17"
    upstream = policy_root / "gr00t_n17/gr00t"
    model_path = policy_root / "model.py"
    process_path = policy_root / "process_data.sh"
    config_path = upstream / "configs/model/gr00t_n1d7.py"
    processor_path = upstream / "model/gr00t_n1d7/processing_gr00t_n1d7.py"
    loss_path = upstream / "model/gr00t_n1d7/gr00t_n1d7.py"

    adapter_text = _read(model_path)
    process_text = _read(process_path)
    config_text = _read(config_path)
    processor_text = _read(processor_path)
    loss_text = _read(loss_path)

    max_state_dim = _parse_int_assignment(config_text, "max_state_dim")
    max_action_dim = _parse_int_assignment(config_text, "max_action_dim")
    action_horizon = _parse_int_assignment(config_text, "action_horizon")

    input_6_plus_1 = (
        re.search(r"_as_1d\([^\n]+,\s*6\)", adapter_text) is not None
        and re.search(r"_as_1d\([^\n]+,\s*1\)", adapter_text) is not None
    )
    output_6_plus_1 = all(
        token in adapter_text
        for token in (
            "left[:6]",
            "left[6:7]",
            "right[:6]",
            "right[6:7]",
        )
    )
    adapter_is_14d_arx = input_6_plus_1 and output_6_plus_1
    process_is_arx_only = (
        "arx_x5)" in process_text
        and "Unsupported env_cfg_type" in process_text
        and re.search(r"(?:spark[_-]?0|tianji_marvin_wuji)\)", process_text, re.IGNORECASE)
        is None
    )

    state_padding = "self.max_state_dim - normalized_states.shape[1]" in processor_text
    action_padding = "self.max_action_dim - normalized_actions.shape[1]" in processor_text
    horizon_mask = "action_mask[action_horizon:] = 0" in processor_text
    dimension_mask = "action_mask[:, action_dim:] = 0" in processor_text
    masked_loss = (
        re.search(r"mse_loss\([^\n]+reduction=[\"']none[\"']\)\s*\*\s*action_mask", loss_text)
        is not None
        and "action_mask.sum()" in loss_text
    )
    masked_padding = all(
        (state_padding, action_padding, horizon_mask, dimension_mask, masked_loss)
    )
    exact_reviewed_envelope = (
        max_state_dim == EXPECTED_MAX_DIM
        and max_action_dim == EXPECTED_MAX_DIM
        and action_horizon == EXPECTED_ACTION_HORIZON
    )
    base_supports_joint54 = (
        exact_reviewed_envelope
        and max_state_dim >= 54
        and max_action_dim >= 54
        and masked_padding
    )

    return {
        "max_state_dim": max_state_dim,
        "max_action_dim": max_action_dim,
        "action_horizon": action_horizon,
        "matches_reviewed_132d_config": exact_reviewed_envelope,
        "state_padding_verified": state_padding,
        "action_padding_verified": action_padding,
        "horizon_mask_verified": horizon_mask,
        "dimension_mask_verified": dimension_mask,
        "masked_loss_verified": masked_loss,
        "masked_padding_verified": masked_padding,
        "base_model_supports_joint54_without_head_resize": base_supports_joint54,
        "adapter_input_is_6_plus_1_per_side": input_6_plus_1,
        "adapter_output_is_6_plus_1_per_side": output_6_plus_1,
        "adapter_is_current_14d_arx_contract": adapter_is_14d_arx,
        "process_script_is_arx_x5_only": process_is_arx_only,
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
        "data_only_compatible_with_current_xpl_adapter": False,
        "expected_spark0_layout": [
            "left_arm:7",
            "left_hand:20",
            "right_arm:7",
            "right_hand:20",
        ],
        "expected_spark0_dim": 54,
        "spark0_has_waist": False,
        "head_resize_required": False,
        "checker_is_read_only": True,
    }

    try:
        report["source"] = _inspect_source(args.source_root)
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        report["source_error"] = f"{type(exc).__name__}: {exc}"
        return _emit(report, "SOURCE_CONTRACT_MISMATCH", EXIT_SOURCE_CONTRACT_MISMATCH)

    try:
        report["local_code"] = _inspect_local_code(repo_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        report["local_code_error"] = f"{type(exc).__name__}: {exc}"
        return _emit(report, "MODEL_ENVELOPE_MISMATCH", EXIT_MODEL_ENVELOPE_MISMATCH)

    if not report["source"]["matches_joint54"]:
        return _emit(report, "SOURCE_CONTRACT_MISMATCH", EXIT_SOURCE_CONTRACT_MISMATCH)

    local = report["local_code"]
    if not local["base_model_supports_joint54_without_head_resize"]:
        report["head_resize_required"] = None
        return _emit(report, "MODEL_ENVELOPE_MISMATCH", EXIT_MODEL_ENVELOPE_MISMATCH)

    current_adapter_gap = (
        local["adapter_is_current_14d_arx_contract"]
        and local["process_script_is_arx_x5_only"]
    )
    if current_adapter_gap:
        report["reason"] = (
            "The local N1.7 base model accepts 54 valid dimensions inside its verified "
            "132D masked envelope, so no head resize is required. The official XPL "
            "adapter still packs 6D arm + 1D gripper per side and process_data.sh is "
            "ARX-only; enabling Spark0 therefore requires adapter/modality changes, "
            "fresh statistics, and a new checkpoint rather than a data-only conversion."
        )
        report["required_adapter_changes"] = [
            "four state/action groups covering 7/20/7/20 with no waist",
            "7D arm and 20D hand packing/unpacking in policy/GR00T_N17/model.py",
            "Spark0 modality/action representations in process_data.sh/config",
            "fresh joint54 statistics, checkpoint, and strict inference validation",
        ]
        return _emit(
            report,
            "XPL_ADAPTER_CHANGE_REQUIRED_BASE_MODEL_CAPABLE",
            EXIT_ADAPTER_CHANGE_REQUIRED,
        )

    report["reason"] = (
        "The official XPL adapter no longer matches the reviewed 14D ARX-only code. "
        "Repeat the end-to-end joint54 compatibility review before converting or training."
    )
    return _emit(report, "ADAPTER_REVIEW_REQUIRED", EXIT_ADAPTER_REVIEW_REQUIRED)


if __name__ == "__main__":
    raise SystemExit(main())
