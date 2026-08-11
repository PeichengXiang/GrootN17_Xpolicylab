# Spark0 joint54 compatibility boundary

Current status: **the GR00T N1.7 base model can represent joint54, but the
official XPolicyLab adapter cannot be made joint54-compatible by a data-only
conversion**.

## What is compatible

Spark0 state and action both use exactly four groups and no waist:

```text
left_arm(7) + left_hand(20) + right_arm(7) + right_hand(20) = 54
```

The vendored N1.7 model is configured with `max_state_dim=132` and
`max_action_dim=132`. Its processor concatenates the valid action groups,
pads them to 132D, and sets the dimension and horizon padding to zero in
`action_mask`. The model multiplies the elementwise action loss by this mask.
Consequently, 54 valid dimensions fit inside the existing 132D envelope; the
remaining 78 dimensions are padding and do not contribute to training loss.

This does **not** require resizing or replacing the state encoder, action
encoder, action decoder, diffusion head, or pretrained checkpoint tensors.
Changing the model's 132D head to 54D would instead break base-checkpoint shape
compatibility.

## What is not compatible today

The official XPolicyLab integration is still ARX-specific:

- `policy/GR00T_N17/model.py` accepts each side as `arm(6) + gripper(1)` and
  slices model output with `[:6]` and `[6:7]`, so its runtime ABI is 14D.
- `policy/GR00T_N17/process_data.sh` only defines the `arx_x5` dataset and a
  two-group `left_arm(7) + right_arm(7)` modality.

Mapping Spark0 to that layout would drop 40 hand joints and mislabel each
seventh arm joint as a scalar gripper. This is semantic data loss, not a valid
format conversion. The current data-only workflow must therefore stop before
dataset generation or training.

## Adapter work required to enable joint54

Enabling Spark0 requires a separate implementation change, followed by fresh
data statistics and a fresh checkpoint:

1. Add a Spark0 branch to `process_data.sh` and describe four non-overlapping
   state/action groups covering all 54 source dimensions. Preserve the source
   packing order; do not add a waist group or a synthetic waist value.
2. Register a four-group modality config. A suitable semantic layout is arms
   as relative `NON_EEF` joint targets and hands as absolute `NON_EEF` joint
   targets. The `action_configs` order must exactly match the modality-key
   order.
3. Change `model.py` observation packing and action unpacking to accept and
   return `7/20/7/20`, without truncation, hand reduction, or arm/hand swaps.
4. Regenerate `meta/stats.json` and `meta/relative_stats.json`, then train a
   new `NEW_EMBODIMENT` checkpoint whose saved processor config and statistics
   use the four joint54 groups.
5. Validate strict inference and the XPolicyLab environment contract before a
   real training or evaluation launch.

The N1.7 model head, `max_state_dim`, `max_action_dim`, and the Spark0 layout
must not be changed as part of that adapter work. Spark0 joint54 contains no
waist; borrowing a fuller humanoid schema would create a different action ABI.

## Executable audit

Run the read-only checker from the repository root:

```bash
/personal/miniconda3/envs/xpolicylab/bin/python \
  scripts/check_groot_joint54_compatibility.py
```

The checker reports the source layout, local N1.7 envelope/masked-padding
evidence, and current XPolicyLab adapter contract. Exit codes are intentional:

- `2`: expected current state — base model supports 54D, adapter changes are
  required, and a data-only conversion is blocked;
- `3`: source is not the expected no-waist joint54 contract;
- `4`: the local N1.7 132D envelope or masked-padding implementation differs
  from the reviewed version;
- `5`: the XPolicyLab adapter changed, so compatibility must be reviewed again.

The checker never converts data, edits the adapter, or starts training.
