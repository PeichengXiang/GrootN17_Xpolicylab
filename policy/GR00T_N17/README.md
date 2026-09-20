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

### EgoVLA raw-action training

The EgoVLA launcher is deliberately fail-closed: it accepts exactly eight
GPUs, global batch size 64, 80,000 optimizer steps, checkpoints every 10,000
steps, and W&B online mode. It recomputes the full 38-D action/state stream
digests before launch and rejects any dataset whose `action` differs from the
same-timestep raw HDF5 `/action` provenance.

```bash
MODEL_ROOT=/path/to/GrootN17
RAW_ROOT=/path/to/EgoVLA_raw_remove_deprecated
DONOR_V30=/path/to/EgoVLA_benchmark_v30
PY="$MODEL_ROOT/policy/GR00T_N17/gr00t_n17/.venv/bin/python"

"$PY" "$MODEL_ROOT/data_scripts/prepare_egovla_v21_template.py" \
  --source-v30 "$DONOR_V30" \
  --source-raw "$RAW_ROOT" \
  --output "$MODEL_ROOT/data/EgoVLA_benchmark_template_v21" \
  --gr00t-root "$MODEL_ROOT/policy/GR00T_N17/gr00t_n17" \
  --modality-config "$MODEL_ROOT/policy/GR00T_N17/configs/ego_h1_inspire_config.py"

"$PY" "$MODEL_ROOT/data_scripts/convert_egovla_to_groot.py" \
  --source-raw "$RAW_ROOT" \
  --source-v21 "$MODEL_ROOT/data/EgoVLA_benchmark_template_v21" \
  --output "$MODEL_ROOT/data/EgoVLA_benchmark_raw_action_v21" \
  --gr00t-root "$MODEL_ROOT/policy/GR00T_N17/gr00t_n17" \
  --modality-config "$MODEL_ROOT/policy/GR00T_N17/configs/ego_h1_inspire_config.py"

EGO_VLA_DATASET_PATH="$MODEL_ROOT/data/EgoVLA_benchmark_raw_action_v21" \
GR00T_BASE_MODEL=/path/to/GR00T-N1.7-3B \
GR00T_COSMOS_MODEL=/path/to/Cosmos-Reason2-2B \
WANDB_MODE=online \
  bash "$MODEL_ROOT/policy/GR00T_N17/scripts/train_egovla_groot_joint38.sh"
```

The v2.1 preparation audit verifies all 5,709 videos, decodes every synthetic
wrist frame, and compares RGB against raw HDF5 samples from all 12 tasks. The
same camera contract is enforced at inference: only
`Humanoid-Insert-And-Unload-Cans-v0` uses real wrists; all other tasks receive
384×384 RGB black wrist images.

## Evaluation

```bash
cd XPolicyLab/policy/GR00T_N17
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_conda_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 <policy_conda_env> <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

### EgoVLA H1-Inspire evaluation

This checkout also contains the 38-D EgoVLA adapter used with the external
EgoVLA benchmark workspace. The benchmark still receives one absolute 50-D H1
target; its integration layer scatters the policy's `left_arm(7),
left_hand(12), right_arm(7), right_hand(12)` fields into that native target.
Use the adapted checkout as the policy root and set the benchmark root
explicitly (the scripts also consume `EGOVLA_WORKSPACE_ROOT` when launched by
the bridge):

```bash
MODEL_ROOT=/personal/xiangpc/0811_Xpolicylab_bench/GrootN17
EGO_ROOT="/personal/xiangpc/EgoVLA benchmark"
CHECKPOINT="$MODEL_ROOT/policy/GR00T_N17/checkpoints/EgoVLA-all_tasks-ego_h1_inspire-joint-38d-raw-action-seed42-20260920/checkpoint-20000"

# Offline adapter/transport smoke; DEBUG_OBS_ENCODED also tests server-side
# image decoding. It does not launch Isaac Sim.
EVAL_MAIN_ROOT="$EGO_ROOT" EVAL_ENV_TYPE=debug DEBUG_OBS_ENCODED=1 \
  bash "$MODEL_ROOT/policy/GR00T_N17/eval.sh" \
  EgoVLA Humanoid-Push-Box-v0 "$CHECKPOINT" ego_h1_inspire joint 0 \
  0 1 "$MODEL_ROOT/policy/GR00T_N17/gr00t_n17" \
  "$EGO_ROOT/.runtime/conda/egovla-isaaclab-1.2.0"

# Real simulator rollout (after reviewing the NVIDIA EULA): export
# ACCEPT_EULA=Y, remove EVAL_ENV_TYPE=debug, and choose the requested task.
```

The policy environment argument may be the GR00T project root, its `.venv`
prefix, or a conda environment. Checkpoints are resolved from an explicit
`checkpoint-*` directory, `model_dir`, or the conventional run directory; the
adapter verifies the 38-D modality groups before serving actions. The current
EgoVLA bridge requires `env_cfg_type=ego_h1_inspire` and `action_type=joint`.
The adapter and training metadata both preserve the released benchmark's
historical `andd then` instruction exactly, so train and inference prompts do
not diverge.

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
