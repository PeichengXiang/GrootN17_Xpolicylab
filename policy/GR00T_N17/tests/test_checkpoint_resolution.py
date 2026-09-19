from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from policy.GR00T_N17.model import (
    _resolve_checkpoint_dir,
    _resolve_cosmos_model,
)


class CheckpointResolutionTest(unittest.TestCase):
    def test_absolute_step_directory_is_loaded_directly(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            checkpoint = Path(raw_dir) / "checkpoint-5000"
            checkpoint.mkdir()
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors.index.json").write_text(
                '{"weight_map": {}}',
                encoding="utf-8",
            )
            with patch(
                "policy.GR00T_N17.model.resolve_checkpoint_root",
                return_value=checkpoint,
            ):
                self.assertEqual(
                    _resolve_checkpoint_dir({"ckpt_name": str(checkpoint)}),
                    checkpoint.resolve(),
                )

    def test_web_cosmos_environment_override_is_honored(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            cosmos = Path(raw_dir)
            (cosmos / "config.json").write_text("{}", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {"GR00T_COSMOS_MODEL": str(cosmos)},
                clear=False,
            ):
                self.assertEqual(
                    _resolve_cosmos_model(
                        {"cosmos_model_path": "nvidia/Cosmos-Reason2-2B"}
                    ),
                    str(cosmos.resolve()),
                )


if __name__ == "__main__":
    unittest.main()
