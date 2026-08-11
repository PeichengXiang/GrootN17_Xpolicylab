# GR00T_N17

**Contributor:** RoboDojo Team | **Paper:** GR00T N1 / N1.5 open foundation model reports | **arXiv:** TBD | **Original code:** https://github.com/NVIDIA/Isaac-GR00T

`GR00T_N17` adapts the NVIDIA Isaac GR00T N1.7 foundation model (base model `nvidia/GR00T-N1.7-3B`) to XPolicyLab/RoboDojo. Integration scripts live at this directory level; the vendored upstream implementation lives in `gr00t_n17/`, with adapter modality configs in `configs/` and helper scripts in `scripts/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

Read `INSTALLATION.md` before first use; it covers setup that `install.sh` cannot fully express, such as external checkpoints, system packages, manual fallback steps, and multi-environment runtime notes. This adapter uses a uv-managed environment.

```bash
cd XPolicyLab/policy/GR00T_N17
bash install.sh
source gr00t_n17/.venv/bin/activate  # or pass `uv` as <policy_env>
```

## Data Processing

```bash
cd XPolicyLab/policy/GR00T_N17
export GR00T_LEROBOT_HOME=/path/to/lerobot/datasets  # required
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: name a 50-episode ablation; point GR00T_SRC_DATASET at the subset dataset first
GR00T_SRC_DATASET=RoboDojo_sim_arx-x5_50ep bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint

# Spark0 joint54: validate the six raw-HDF5 tasks without writing data
GR00T_PROCESS_PREVIEW=1 bash process_data.sh \
  Spark0_bench groot_n17_lerobotV21_joint54 tianji_marvin_wuji joint

# Spark0 joint54: direct raw HDF5 -> LeRobot v2.1 conversion
bash process_data.sh \
  Spark0_bench groot_n17_lerobotV21_joint54 tianji_marvin_wuji joint
```

The ARX source dataset defaults to `RoboDojo_sim_arx-x5_v30`;
`expert_data_num` is accepted for ARX compatibility only and is not applied for
episode subsetting. The `tianji_marvin_wuji` branch instead reads raw Spark0
HDF5 from `/mnt/xspark-data/tjy/spark0_bench` (override with
`GR00T_SPARK0_HDF5_ROOT`), treats the optional fifth argument as episodes per
task, and refuses to reuse `sim_6tasks_lerobot` or overwrite an existing target.
Its physical vector layout is `left_arm(7), left_hand(20), right_arm(7),
right_hand(20)` at 25 Hz; the tracked processor config uses official modality
order `left_arm, right_arm, left_hand, right_hand`.

Conversion stays in a private staging directory until the tracked independent
auditor has compared every Parquet state/action frame with the raw HDF5,
checked every video with `ffprobe`, validated absolute/relative statistics, and
loaded the first and last episode of every task through the official GR00T
loader. Any failed check blocks the atomic publish. A successful audit writes
`meta/spark0_joint54_audit.json`, binding the result to the dataset tree,
manifest, modality config, and auditor version; the Spark0 training wrapper
verifies that marker and identity before launch.

To recheck an already audited dataset without repeating video decoding:

```bash
PYTHONPATH="../..:gr00t_n17${PYTHONPATH:+:$PYTHONPATH}" \
  gr00t_n17/.venv/bin/python scripts/audit_spark0_groot_v21.py \
  --dataset "$GR00T_LEROBOT_HOME/Spark0_bench-groot_n17_lerobotV21_joint54-tianji_marvin_wuji-joint" \
  --config-path configs/tianji_marvin_wuji_config.py \
  --verify-marker-only --require-full
```

## Training

```bash
cd XPolicyLab/policy/GR00T_N17
export GR00T_LEROBOT_HOME=/path/to/lerobot/datasets  # required

bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (use gpu_id 0,1,2,3 for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Spark0 joint54: preview the fixed 8-GPU recipe without launching
GR00T_TRAIN_PREVIEW=1 bash scripts/train_spark0_groot_joint54.sh

# Spark0 joint54: 8 GPUs, global batch 64, 100k steps, save every 5k, W&B online
bash scripts/train_spark0_groot_joint54.sh
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name, the full run-directory name, or a path to a checkpoint directory — or set `model_dir` in `deploy.yml` to a directory relative to `policy/GR00T_N17/`. The base models default to `nvidia/GR00T-N1.7-3B` and `nvidia/Cosmos-Reason2-2B` (override with `GR00T_BASE_MODEL` / `GR00T_COSMOS_MODEL`). The process count is inferred from a comma-separated `gpu_id` (`NUM_GPUS` override), and `GLOBAL_BATCH_SIZE` / `MAX_STEPS` can be exported to tune the run, e.g. `GLOBAL_BATCH_SIZE=640 MAX_STEPS=60000 bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1,2,3,4,5,6,7`.

The Spark0 wrapper fixes the requested recipe and delegates to the unchanged
`train.sh`. It refuses an existing output directory and refuses to use a prior
GR00T fine-tuned checkpoint as its base, so the joint54 run cannot implicitly
resume the ARX checkpoint. It requires the complete 600-episode / 151,410-frame
dataset with absolute and relative statistics, uses the repository's local
GR00T/Cosmos symlinks with Hugging Face offline, and checks that all requested
GPUs are idle before a real launch. `GR00T_TRAIN_PREVIEW=1` performs the other
checks and prints the full run identity without launching training.

## Evaluation

```bash
cd XPolicyLab/policy/GR00T_N17
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `embodiment_tag`, `checkpoint_num`, `model_dir` (optional checkpoint directory relative to `policy/GR00T_N17/`; when set, it bypasses `checkpoints/<ckpt_name>`), `cosmos_model_path` (Hugging Face repo id, or local Cosmos directory relative to `policy/GR00T_N17/`), `default_prompt`, `policy_uv_env_path`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `GR00T_LEROBOT_HOME` | LeRobot dataset root used by `process_data.sh` and `train.sh` (required). |
| `GR00T_SRC_DATASET` | Source dataset for data processing; defaults to `RoboDojo_sim_arx-x5_v30` for `arx_x5`. |
| `GR00T_SPARK0_HDF5_ROOT` | Raw six-task Spark0 HDF5 root; defaults to `/mnt/xspark-data/tjy/spark0_bench`. |
| `GR00T_LEROBOT_V21_PYTHON` | Optional LeRobot 0.3.3 conversion Python; defaults to the server's `dexora_1b` environment. |
| `GR00T_AUDIT_FFPROBE_WORKERS` | Parallel `ffprobe` workers for the mandatory pre-publish Spark0 audit; defaults to `8`. |
| `GR00T_BASE_MODEL` | Base GR00T model path or Hugging Face id used by `train.sh`. |
| `GR00T_COSMOS_MODEL` | Cosmos model path or Hugging Face id used by `train.sh`. |

Additional optional overrides: `GR00T_ROOT`, `GR00T_VIDEO_BACKEND`.

## Notes

- The policy-environment argument of the eval scripts accepts a conda env name, `uv`, or a uv project path.
- For data-size ablations, create or select a subset source dataset with `GR00T_SRC_DATASET` and encode the subset in `ckpt_name` such as `stack_bowls_50ep`.
- See `../../scripts/GR00T_N17_SPARK0_JOINT54_SUPPORT.md` for the reviewed
  physical/modality ordering, unchanged 132D envelope, safeguards, and audit
  exit codes.
