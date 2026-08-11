# GR00T N1.7 Spark0 joint54 support

Current status: **the XPolicyLab GR00T N1.7 adapter supports Spark0 joint54
without changing the vendored N1.7 model or pretrained tensor shapes**.

## Contract

The raw HDF5 and LeRobot vectors keep the physical side-grouped layout:

```text
left_arm[0:7] + left_hand[7:27] + right_arm[27:34] + right_hand[34:54]
```

The N1.7 processor consumes the same ranges in its reviewed modality order:

```text
left_arm + right_arm + left_hand + right_hand
```

Arm actions are relative joint targets and hand actions are absolute joint
targets. Spark0 contains no waist group. The adapter gets the `7/20` dimensions
from XPolicyLab robot metadata and reorders groups by name; it does not truncate,
reduce, or swap joints. The existing ARX `6+1` path remains two combined groups
(`left_arm`, `right_arm`).

The vendored N1.7 model remains at `max_state_dim=132`, `max_action_dim=132`,
and `action_horizon=40`. Its processor pads valid dimensions and its masked loss
ignores the padding, so the 54D embodiment needs no head resize.

## Data preview and conversion

The Spark0 path reads raw HDF5 directly from
`/mnt/xspark-data/tjy/spark0_bench` by default. It selects exactly six tasks,
uses all 100 episodes per task by default, preserves 25 Hz and 480x640 RGB, and
publishes a fresh LeRobot v2.1 dataset only after schema/statistics validation.
It refuses `sim_6tasks_lerobot` as an input and refuses to overwrite an existing
output dataset. The full-dataset training gate requires exactly 600 episodes,
151,410 frames, six tasks, and both absolute and relative statistics.

```bash
cd policy/GR00T_N17
export GR00T_LEROBOT_HOME=/path/to/lerobot/datasets

GR00T_PROCESS_PREVIEW=1 bash process_data.sh \
  Spark0_bench groot_n17_lerobotV21_joint54 tianji_marvin_wuji joint

bash process_data.sh \
  Spark0_bench groot_n17_lerobotV21_joint54 tianji_marvin_wuji joint
```

Set `GR00T_SPARK0_HDF5_ROOT` only to select another raw-HDF5 root. Set
`GR00T_LEROBOT_V21_PYTHON` only to select another environment that has exactly
LeRobot 0.3.3; the default is `/personal/miniconda3/envs/dexora_1b/bin/python`.

## Training preview and launch

The dedicated recipe uses a new `groot_n17_lerobotV21_joint54` run identity,
the repository's local N1.7 and Cosmos symlinks in Hugging Face offline mode,
8 GPUs, global batch 64, 100,000 steps, checkpoint interval 5,000, and W&B
online. It refuses an existing output directory, a fine-tuned GR00T checkpoint
as the base, an incomplete full dataset, or GPUs with existing compute jobs, so
it cannot implicitly resume the ARX run or collide with another training job.

```bash
GR00T_TRAIN_PREVIEW=1 bash scripts/train_spark0_groot_joint54.sh

# Launch only after the data conversion and preview gates pass.
bash scripts/train_spark0_groot_joint54.sh
```

Both wrappers prepend the current repository and vendored GR00T root to
`PYTHONPATH`, then assert the resolved `XPolicyLab` and `gr00t` module paths.
This prevents the shared environment's older editable mappings from silently
loading another checkout.

## Executable audit

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/policy/GR00T_N17/gr00t_n17" \
  policy/GR00T_N17/gr00t_n17/.venv/bin/python \
  scripts/check_groot_joint54_compatibility.py
```

Exit `0` means the raw source, unchanged 132D envelope, adapter, configs,
conversion safeguards, and training recipe match this contract. Exit `3`
denotes a source mismatch, `4` an upstream envelope/masking mismatch, and `5`
an adapter/config/recipe review failure. The checker is read-only.
