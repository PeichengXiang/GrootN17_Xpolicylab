#!/usr/bin/env python3
"""Fail-fast compatibility audit for Spark0 joint54 and official XPL GR00T N1.7.

This intentionally does not generate a lossy 14D dataset.  The XPolicyLab
adapter is the limiting ABI and the user requested data-only adaptation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py


DEFAULT_SOURCE = Path("/mnt/xspark-data/tjy/spark0_bench")


def width(handle: h5py.File, prefix: str) -> int:
    keys = (
        f"{prefix}/left_arm_joint_states",
        f"{prefix}/left_ee_joint_states",
        f"{prefix}/right_arm_joint_states",
        f"{prefix}/right_ee_joint_states",
    )
    return sum(int(handle[key].shape[1]) for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "policy/GR00T_N17/model.py"
    process_path = repo_root / "policy/GR00T_N17/process_data.sh"
    episodes = sorted(args.source_root.glob("*/tianji_marvin_wuji/data/episode_*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"no source episodes below {args.source_root}")

    with h5py.File(episodes[0], "r") as handle:
        state_width = width(handle, "state")
        action_width = width(handle, "action")

    model_text = model_path.read_text(encoding="utf-8")
    process_text = process_path.read_text(encoding="utf-8")
    six_dim_calls = len(re.findall(r"_as_1d\([^\n]+,\s*6\)", model_text))
    one_dim_calls = len(re.findall(r"_as_1d\([^\n]+,\s*1\)", model_text))
    official_14d = six_dim_calls >= 1 and one_dim_calls >= 1
    arx_only = "arx_x5" in process_text and "Unsupported env_cfg_type" in process_text

    report = {
        "compatible": False,
        "source_episode": str(episodes[0]),
        "source_state_dim": state_width,
        "source_action_dim": action_width,
        "source_layout": ["left_arm:7", "left_hand:20", "right_arm:7", "right_hand:20"],
        "official_xpl_adapter_layout": ["left_arm:6", "left_gripper:1", "right_arm:6", "right_gripper:1"],
        "official_xpl_adapter_dim": 14,
        "model_has_6_plus_1_contract": official_14d,
        "process_script_is_arx_x5_only": arx_only,
        "reason": (
            "A data-only conversion cannot map two 20D dexterous hands and 7D arms "
            "onto the official 6D-arm + 1D-gripper-per-side ABI without dropping or "
            "mislabeling joints. The XPolicyLab adapter/dataloader must remain unchanged."
        ),
    }
    print(json.dumps(report, indent=2))

    if state_width != 54 or action_width != 54:
        raise RuntimeError("source is not the expected joint54 ABI")
    if not official_14d or not arx_only:
        raise RuntimeError("official adapter changed; repeat the compatibility review")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
