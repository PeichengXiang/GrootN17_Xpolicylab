from __future__ import annotations

import errno
import unittest
from unittest.mock import patch

import numpy as np
from policy.GR00T_N17.model import (
    EGO_VLA_REAL_WRIST_PROMPT,
    _encode_observation,
    _extract_prompt,
    _gr00t_action_to_env,
    _gr00t_group_dims,
    _pack_state_groups,
    _resolve_robot_action_dim_info,
)


def _state(
    left_arm: np.ndarray,
    left_ee: np.ndarray,
    right_arm: np.ndarray,
    right_ee: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "state": {
            "left_arm_joint_state": left_arm,
            "left_ee_joint_state": left_ee,
            "right_arm_joint_state": right_arm,
            "right_ee_joint_state": right_ee,
        }
    }


class JointSchemaTest(unittest.TestCase):
    def test_benchmark_instruction_is_preserved_exactly(self) -> None:
        prompt = _extract_prompt(
            {
                "instruction": (
                    "Insert the left can into the slot and insert the right can into the slot, "
                    "unload the left cans andd then unload the right cans"
                )
            },
            "fallback",
        )
        self.assertEqual(prompt, EGO_VLA_REAL_WRIST_PROMPT)

    @staticmethod
    def _egovla_observation(prompt: str, shape=(384, 384, 3)) -> dict:
        head = np.full(shape, 17, dtype=np.uint8)
        left = np.full(shape, 31, dtype=np.uint8)
        right = np.full(shape, 47, dtype=np.uint8)
        observation = _state(
            np.zeros(7, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
            np.zeros(7, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        )
        observation.update(
            {
                "instruction": prompt,
                "vision": {
                    "cam_head": {"color": head},
                    "cam_left_wrist": {"color": left},
                    "cam_right_wrist": {"color": right},
                },
            }
        )
        return observation

    def test_head_only_egovla_task_forces_black_wrists(self) -> None:
        encoded = _encode_observation(
            self._egovla_observation("Close the opened drawer"),
            "unused",
            "joint",
            {"arm_dim": [7, 7], "ee_dim": [12, 12]},
            "ego_h1_inspire",
            "Humanoid-Close-Drawer-v0",
        )
        self.assertTrue(np.all(encoded["video"]["left_wrist"] == 0))
        self.assertTrue(np.all(encoded["video"]["right_wrist"] == 0))
        self.assertTrue(np.all(encoded["video"]["front"] == 17))

    def test_three_view_egovla_task_preserves_real_wrists(self) -> None:
        encoded = _encode_observation(
            self._egovla_observation(EGO_VLA_REAL_WRIST_PROMPT),
            "unused",
            "joint",
            {"arm_dim": [7, 7], "ee_dim": [12, 12]},
            "ego_h1_inspire",
            "Humanoid-Insert-And-Unload-Cans-v0",
        )
        self.assertTrue(np.all(encoded["video"]["left_wrist"] == 31))
        self.assertTrue(np.all(encoded["video"]["right_wrist"] == 47))

    def test_egovla_camera_and_prompt_contract_fail_closed(self) -> None:
        metadata = {"arm_dim": [7, 7], "ee_dim": [12, 12]}
        with self.assertRaisesRegex(ValueError, "task/prompt mismatch"):
            _encode_observation(
                self._egovla_observation("Perform the task"),
                "unused",
                "joint",
                metadata,
                "ego_h1_inspire",
                "Humanoid-Close-Drawer-v0",
            )
        with self.assertRaisesRegex(ValueError, "task/prompt mismatch"):
            _encode_observation(
                self._egovla_observation("Open the closed drawer"),
                "unused",
                "joint",
                metadata,
                "ego_h1_inspire",
                "Humanoid-Close-Drawer-v0",
            )
        with self.assertRaisesRegex(ValueError, "RGB uint8"):
            _encode_observation(
                self._egovla_observation("Close the opened drawer", shape=(256, 256, 3)),
                "unused",
                "joint",
                metadata,
                "ego_h1_inspire",
                "Humanoid-Close-Drawer-v0",
            )

    def test_arx_metadata_resolver_regression(self) -> None:
        metadata = _resolve_robot_action_dim_info("arx_x5")
        self.assertEqual(metadata["arm_dim"], [6, 6])
        self.assertEqual(metadata["ee_dim"], [1, 1])
        self.assertEqual(
            _gr00t_group_dims(metadata),
            [("left_arm", 7), ("right_arm", 7)],
        )

    def test_metadata_resolver_does_not_mask_other_missing_files(self) -> None:
        missing = FileNotFoundError(errno.ENOENT, "missing", "/tmp/shared_robot_info.json")
        with patch(
            "policy.GR00T_N17.model.get_robot_action_dim_info",
            side_effect=missing,
        ):
            with self.assertRaises(FileNotFoundError) as caught:
                _resolve_robot_action_dim_info("arx_x5")
        self.assertEqual(caught.exception.filename, missing.filename)

    def test_arx_6_plus_1_regression(self) -> None:
        metadata = {"arm_dim": [6, 6], "ee_dim": [1, 1]}
        left_arm = np.arange(6, dtype=np.float32)
        left_ee = np.array([6], dtype=np.float32)
        right_arm = np.arange(10, 16, dtype=np.float32)
        right_ee = np.array([16], dtype=np.float32)

        groups = _pack_state_groups(
            _state(left_arm, left_ee, right_arm, right_ee),
            "joint",
            metadata,
        )
        self.assertEqual(list(groups), ["left_arm", "right_arm"])
        np.testing.assert_array_equal(groups["left_arm"], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(groups["right_arm"], np.arange(10, 17, dtype=np.float32))

        action = {
            "left_arm": groups["left_arm"][None, None, :],
            "right_arm": groups["right_arm"][None, None, :],
        }
        unpacked = _gr00t_action_to_env(action, "joint", metadata)
        self.assertEqual(len(unpacked), 1)
        np.testing.assert_array_equal(unpacked[0]["left_arm_joint_state"], left_arm)
        np.testing.assert_array_equal(unpacked[0]["left_ee_joint_state"], left_ee)
        np.testing.assert_array_equal(unpacked[0]["right_arm_joint_state"], right_arm)
        np.testing.assert_array_equal(unpacked[0]["right_ee_joint_state"], right_ee)

    def test_spark0_7_plus_20_round_trip(self) -> None:
        metadata = {"arm_dim": [7, 7], "ee_dim": [20, 20]}
        left_arm = np.arange(0, 7, dtype=np.float32)
        left_hand = np.arange(100, 120, dtype=np.float32)
        right_arm = np.arange(200, 207, dtype=np.float32)
        right_hand = np.arange(300, 320, dtype=np.float32)

        groups = _pack_state_groups(
            _state(left_arm, left_hand, right_arm, right_hand),
            "joint",
            metadata,
        )
        self.assertEqual(
            list(groups),
            ["left_arm", "right_arm", "left_hand", "right_hand"],
        )
        np.testing.assert_array_equal(groups["left_arm"], left_arm)
        np.testing.assert_array_equal(groups["right_arm"], right_arm)
        np.testing.assert_array_equal(groups["left_hand"], left_hand)
        np.testing.assert_array_equal(groups["right_hand"], right_hand)

        action = {name: value[None, None, :] for name, value in groups.items()}
        unpacked = _gr00t_action_to_env(action, "joint", metadata)
        self.assertEqual(len(unpacked), 1)
        np.testing.assert_array_equal(unpacked[0]["left_arm_joint_state"], left_arm)
        np.testing.assert_array_equal(unpacked[0]["left_ee_joint_state"], left_hand)
        np.testing.assert_array_equal(unpacked[0]["right_arm_joint_state"], right_arm)
        np.testing.assert_array_equal(unpacked[0]["right_ee_joint_state"], right_hand)

    def test_spark0_group_dimensions_are_processor_ordered(self) -> None:
        self.assertEqual(
            _gr00t_group_dims({"arm_dim": [7, 7], "ee_dim": [20, 20]}),
            [
                ("left_arm", 7),
                ("right_arm", 7),
                ("left_hand", 20),
                ("right_hand", 20),
            ],
        )


if __name__ == "__main__":
    unittest.main()
