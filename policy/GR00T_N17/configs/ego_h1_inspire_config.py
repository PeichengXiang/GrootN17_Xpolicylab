"""GR00T N1.7 modality definition for the EgoVLA H1-Inspire benchmark.

EgoVLA keeps the manipulation subspace of the 50-DoF H1-Inspire articulation:
7 joints per arm and 12 joints per hand.  The two arm groups are represented
as relative joint actions; hand joints remain absolute, matching the existing
GR00T humanoid/Spark0 convention and the generated ``relative_stats.json``.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


ego_h1_inspire_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        # The loader maps these pretrained/model-facing names positionally to
        # cam_head/cam_left_wrist/cam_right_wrist in meta/modality.json.
        modality_keys=["front", "left_wrist", "right_wrist"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    ego_h1_inspire_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
