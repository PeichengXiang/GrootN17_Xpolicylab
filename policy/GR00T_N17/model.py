from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.checkpoint_io import network_checkpoint_view
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

_POLICY_DIR = Path(__file__).resolve().parent
_GR00T_ROOT = _POLICY_DIR / "gr00t_n17"
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"
_ENV_CFG_DIR = (
    Path(get_robot_action_dim_info.__code__.co_filename).resolve().parent / "../../env_cfg"
).resolve()
_ROBOT_INFO_PATH = _POLICY_DIR.parents[1] / "utils" / "robot" / "_robot_info.json"

if str(_GR00T_ROOT) not in sys.path:
    sys.path.insert(0, str(_GR00T_ROOT))

from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: E402
from gr00t.policy import Gr00tPolicy  # noqa: E402

VIDEO_KEY_CANDIDATES = {
    "front": ["cam_head", "cam_high", "head_camera", "top_camera"],
    "left_wrist": ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"],
    "right_wrist": ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"],
}

EGO_VLA_RAW_IMAGE_SHAPE = (384, 384, 3)
EGO_VLA_REAL_WRIST_PROMPT = (
    "Insert the left can into the slot and insert the right can into the slot, "
    "unload the left cans andd then unload the right cans"
)
EGO_VLA_TASK_PROMPTS = frozenset(
    {
        "pour balls in cup into bowl",
        "push box to the marker",
        "Put sprite cans to the left box, and orange cans to the right box",
        "Insert cans into the boxes",
        "Close the opened drawer",
        "Open the closed drawer",
        EGO_VLA_REAL_WRIST_PROMPT,
        "Flip the mug",
        "unload the right cans and then unload the left cans",
        "put can on the saucer",
        "Open the drawer, and Put can on the saucer",
        "open the laptop",
    }
)


def _validate_robot_action_dim_info(
    env_cfg_type: str,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"Robot metadata for {env_cfg_type!r} must be an object, got {type(value).__name__}."
        )

    validated = dict(value)
    for field in ("arm_dim", "ee_dim"):
        dimensions = value.get(field)
        if not isinstance(dimensions, list) or len(dimensions) != 2:
            raise ValueError(
                f"Robot metadata for {env_cfg_type!r} must define {field} as two dimensions, "
                f"got {dimensions!r}."
            )
        if any(
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            for dimension in dimensions
        ):
            raise ValueError(
                f"Robot metadata for {env_cfg_type!r} has invalid {field}: {dimensions!r}."
            )
        validated[field] = list(dimensions)
    return validated


def _resolve_robot_action_dim_info(env_cfg_type: str) -> dict[str, Any]:
    """Resolve the normal env config first, then the tracked robot schema by name."""
    try:
        metadata = get_robot_action_dim_info(env_cfg_type)
    except FileNotFoundError as exc:
        expected_missing_path = (_ENV_CFG_DIR / f"{env_cfg_type}.yml").resolve()
        actual_missing_path = Path(exc.filename).resolve() if exc.filename else None
        if actual_missing_path != expected_missing_path:
            raise
        with open(_ROBOT_INFO_PATH, encoding="utf-8") as file:
            tracked_robot_info = json.load(file)
        if not isinstance(tracked_robot_info, dict):
            raise TypeError(f"Tracked robot metadata must be an object: {_ROBOT_INFO_PATH}")
        if env_cfg_type not in tracked_robot_info:
            raise KeyError(
                f"Robot {env_cfg_type!r} is absent from tracked metadata {_ROBOT_INFO_PATH}."
            )
        metadata = tracked_robot_info[env_cfg_type]
        warnings.warn(
            f"Environment config {env_cfg_type!r} was not found; using the tracked robot "
            f"schema in {_ROBOT_INFO_PATH}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return _validate_robot_action_dim_info(env_cfg_type, metadata)


def _load_modality_config(env_cfg_type: str) -> None:
    config_path = _POLICY_DIR / "configs" / f"{env_cfg_type}_config.py"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Modality config not found: {config_path}. "
            f"Run process_data.sh for env_cfg_type={env_cfg_type} first."
        )
    spec = importlib.util.spec_from_file_location(f"gr00t_modality_{env_cfg_type}", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load modality config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _extract_step_number(value: Any) -> int | None:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else None


DEFAULT_COSMOS_MODEL_REPO = "nvidia/Cosmos-Reason2-2B"


def _resolve_relative_path(raw_path: str | Path, base_dir: Path) -> Path:
    """Resolve a deploy.yml path relative to base_dir."""
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        raise ValueError(
            f"Absolute paths are not supported: {path}. "
            f"Use a path relative to {base_dir} or set it in deploy.yml."
        )
    return (base_dir / path).resolve()


def _is_hf_repo_id(value: str) -> bool:
    if value.startswith((".", "/")) or "://" in value:
        return False
    parts = value.split("/")
    return len(parts) >= 2 and all(parts)


def _resolve_cosmos_model(model_cfg: dict[str, Any]) -> str:
    """Return HuggingFace repo id or a local path for Cosmos (processor backbone)."""
    raw_path = (
        os.environ.get("GR00T_COSMOS_MODEL", "").strip()
        or model_cfg.get("cosmos_model_path")
    )
    if raw_path is None or raw_path == "":
        return DEFAULT_COSMOS_MODEL_REPO

    raw = str(raw_path)
    for candidate in (
        _POLICY_DIR / raw,
        _CHECKPOINTS_DIR / raw,
        _POLICY_DIR / "checkpoints" / raw,
    ):
        if (candidate / "config.json").is_file():
            return str(candidate.resolve())

    if _is_hf_repo_id(raw):
        return raw

    return str(_resolve_relative_path(raw, _POLICY_DIR))


@contextmanager
def _override_processor_cosmos_model(checkpoint_dir: Path, cosmos_model: str) -> Iterator[None]:
    """Replace baked-in absolute Cosmos paths in processor_config.json during load."""
    config_path = checkpoint_dir / "processor_config.json"
    if not config_path.is_file():
        yield
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    processor_kwargs = data.setdefault("processor_kwargs", {})
    previous = processor_kwargs.get("model_name")
    if previous == cosmos_model:
        yield
        return

    processor_kwargs["model_name"] = cosmos_model
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    try:
        yield
    finally:
        if previous is not None:
            processor_kwargs["model_name"] = previous
        else:
            processor_kwargs.pop("model_name", None)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def _device_is_cpu(device: Any) -> bool:
    """Return whether a model device denotes CPU execution."""
    return str(device).strip().lower().split(":", 1)[0] == "cpu"


@contextmanager
def _cpu_checkpoint_view(checkpoint_dir: Path, device: Any) -> Iterator[Path]:
    """Provide a copy-on-write checkpoint view for CPU debug/eval.

    GR00T-N1.7 checkpoints are trained with FlashAttention2 enabled.  The
    upstream loader reads that setting from ``config.json`` and fails before
    model construction on a CPU-only debug worker.  A temporary directory
    containing symlinks to the (large) weight files lets us switch attention to
    SDPA without mutating the real checkpoint or duplicating tens of GB.
    GPU evaluation keeps the original checkpoint path byte-for-byte intact.
    """
    if not _device_is_cpu(device):
        yield checkpoint_dir
        return

    config_path = checkpoint_dir / "config.json"
    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError):
        # Let the normal loader report a more specific checkpoint error.
        yield checkpoint_dir
        return

    if not config.get("use_flash_attention") and config.get("attn_implementation") != "flash_attention_2":
        yield checkpoint_dir
        return

    with tempfile.TemporaryDirectory(prefix="gr00t_cpu_checkpoint_") as raw_view:
        view = Path(raw_view)
        for entry in checkpoint_dir.iterdir():
            destination = view / entry.name
            if entry.name == "config.json":
                cpu_config = dict(config)
                cpu_config["use_flash_attention"] = False
                if cpu_config.get("attn_implementation") == "flash_attention_2":
                    cpu_config["attn_implementation"] = "sdpa"
                destination.write_text(
                    json.dumps(cpu_config, indent=2),
                    encoding="utf-8",
                )
            elif entry.name == "processor_config.json":
                # This file is also overridden for the local Cosmos path; copy
                # it so that nested context managers never edit the source.
                shutil.copy2(entry, destination)
            else:
                destination.symlink_to(entry, target_is_directory=entry.is_dir())
        yield view


def _is_loadable_checkpoint_dir(path: Path) -> bool:
    """Return whether path is already a complete HF-style step directory."""

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (path / "model.safetensors").is_file() or (
        path / "model.safetensors.index.json"
    ).is_file()


def _resolve_checkpoint_dir(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_dir key > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    root = resolve_checkpoint_root(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_dir",),
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    # Web and the native CLI pass an absolute checkpoint-<step> path so the
    # selected epoch remains exact even when newer checkpoints appear later.
    if _is_loadable_checkpoint_dir(root):
        return root.resolve()

    search_roots = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir() and not child.name.startswith("checkpoint-"):
            search_roots.append(child)

    candidates = []
    for search_root in search_roots:
        candidates.extend(sorted(search_root.glob("checkpoint-*"), key=lambda p: p.name))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint-* directories under {root}")

    checkpoint_num = model_cfg.get("checkpoint_num")
    if checkpoint_num in (None, "last"):
        return max(candidates, key=lambda p: _extract_step_number(p.name) or -1)

    desired = _extract_step_number(checkpoint_num)
    if desired is not None:
        for candidate in candidates:
            if _extract_step_number(candidate.name) == desired:
                return candidate.resolve()

    explicit = root / f"checkpoint-{checkpoint_num}"
    if explicit.is_dir():
        return explicit.resolve()
    for search_root in search_roots:
        nested = search_root / f"checkpoint-{checkpoint_num}"
        if nested.is_dir():
            return nested.resolve()

    raise FileNotFoundError(
        f"Checkpoint step {checkpoint_num!r} not found under {root}. "
        f"Available: {[p.name for p in candidates]}"
    )


def _ensure_hwc_uint8(image: Any) -> np.ndarray:

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image
    if image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

    raise ValueError(f"Unsupported image shape: {image.shape}")


def _extract_image_value(observation: dict[str, Any], candidate_names: list[str]) -> Any:
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "colors", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def _extract_image(observation: dict[str, Any], candidate_names: list[str]) -> np.ndarray:
    return _ensure_hwc_uint8(_extract_image_value(observation, candidate_names))


def _to_rgb_hwc(image: np.ndarray) -> np.ndarray:
    """XPolicyLab obs images are RGB; match LeRobot video training."""
    return _ensure_hwc_uint8(image)


def _require_egovla_rgb(image: Any, camera_name: str) -> np.ndarray:
    """Enforce the decoded EgoVLA runtime boundary: HWC uint8 RGB 384x384."""
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != EGO_VLA_RAW_IMAGE_SHAPE:
        raise ValueError(
            f"EgoVLA {camera_name} must be RGB uint8 {EGO_VLA_RAW_IMAGE_SHAPE}, "
            f"got shape={array.shape}, dtype={array.dtype}"
        )
    return np.ascontiguousarray(array)


def _as_1d(value: Any, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != length:
        raise ValueError(f"Expected length {length}, got {arr.shape}")
    return arr


def _extract_prompt(observation: dict[str, Any], default_prompt: str) -> str:
    for key in ("instruction", "instructions"):
        if key not in observation:
            continue
        value = observation[key]
        if isinstance(value, dict):
            general = value.get("general")
            if isinstance(general, list) and general:
                first = general[0]
                if isinstance(first, dict):
                    conversations = first.get("conversations", [])
                    for turn in conversations:
                        if turn.get("from") == "human" and turn.get("value"):
                            text = str(turn["value"])
                            marker = "Generate robot actions for the task:\n"
                            if marker in text:
                                text = text.split(marker, 1)[1]
                            return text.replace(" /no_cot", "")
        if isinstance(value, list):
            if not value:
                continue
            value = value[0]
        if isinstance(value, str) and value:
            return value
    return str(default_prompt)


def _physical_group_dims(robot_action_dim_info: dict[str, Any]) -> list[tuple[str, int]]:
    arm_dims = list(robot_action_dim_info.get("arm_dim", []))
    ee_dims = list(robot_action_dim_info.get("ee_dim", []))
    if len(arm_dims) != 2 or len(ee_dims) != 2:
        raise ValueError(
            "GR00T_N17 XPolicyLab integration requires a dual-arm robot; "
            f"got arm_dim={arm_dims}, ee_dim={ee_dims}."
        )

    group_dims: list[tuple[str, int]] = []
    for side, arm_dim, ee_dim in zip(("left", "right"), arm_dims, ee_dims):
        if not isinstance(arm_dim, int) or not isinstance(ee_dim, int):
            raise TypeError(
                f"Robot dimensions must be integers, got arm={arm_dim!r}, ee={ee_dim!r}."
            )
        if arm_dim <= 0 or ee_dim <= 0:
            raise ValueError(f"Robot dimensions must be positive, got arm={arm_dim}, ee={ee_dim}.")

        if ee_dim == 1:
            # Preserve the existing ARX checkpoint ABI exactly: arm + scalar
            # gripper live in one group per side.
            group_dims.append((f"{side}_arm", arm_dim + ee_dim))
        else:
            group_dims.extend(
                [
                    (f"{side}_arm", arm_dim),
                    (f"{side}_hand", ee_dim),
                ]
            )
    return group_dims


def _gr00t_group_dims(robot_action_dim_info: dict[str, Any]) -> list[tuple[str, int]]:
    physical_groups = _physical_group_dims(robot_action_dim_info)
    if all(name in {"left_arm", "right_arm"} for name, _ in physical_groups):
        return physical_groups

    by_name = dict(physical_groups)
    modality_order = ("left_arm", "right_arm", "left_hand", "right_hand")
    if set(by_name) != set(modality_order):
        raise ValueError(f"Unsupported GR00T modality groups: {list(by_name)}")
    return [(name, by_name[name]) for name in modality_order]


def _pack_state_groups(
    observation: dict[str, Any],
    action_type: str,
    robot_action_dim_info: dict[str, Any],
) -> dict[str, np.ndarray]:
    if action_type != "joint":
        raise ValueError(
            "GR00T_N17 is configured for joint-space relative actions "
            f"(action_type=joint), got {action_type!r}."
        )

    physical_groups = _physical_group_dims(robot_action_dim_info)
    total_dim = sum(dim for _, dim in physical_groups)
    packed = _as_1d(
        pack_robot_state(
            observation,
            action_type=action_type,
            robot_action_dim_info=robot_action_dim_info,
            source_type="obs",
        ),
        total_dim,
    ).astype(np.float32)

    groups_by_name: dict[str, np.ndarray] = {}
    offset = 0
    for group_name, group_dim in physical_groups:
        groups_by_name[group_name] = packed[offset : offset + group_dim]
        offset += group_dim
    return {
        group_name: groups_by_name[group_name]
        for group_name, _ in _gr00t_group_dims(robot_action_dim_info)
    }


def _encode_observation(
    obs: dict[str, Any],
    default_prompt: str,
    action_type: str,
    robot_action_dim_info: dict[str, Any],
    env_cfg_type: str = "",
) -> dict[str, Any]:
    prompt = _extract_prompt(obs, default_prompt)
    if env_cfg_type == "ego_h1_inspire":
        if prompt not in EGO_VLA_TASK_PROMPTS:
            raise ValueError(
                "EgoVLA inference requires an exact benchmark-registry instruction; "
                f"got {prompt!r}"
            )
        front = _require_egovla_rgb(
            _extract_image_value(obs, VIDEO_KEY_CANDIDATES["front"]), "front"
        )
        if prompt == EGO_VLA_REAL_WRIST_PROMPT:
            left_wrist = _require_egovla_rgb(
                _extract_image_value(obs, VIDEO_KEY_CANDIDATES["left_wrist"]),
                "left_wrist",
            )
            right_wrist = _require_egovla_rgb(
                _extract_image_value(obs, VIDEO_KEY_CANDIDATES["right_wrist"]),
                "right_wrist",
            )
        else:
            # These 11 benchmark tasks are head-only in training.  Ignore any
            # simulator wrist payload and reproduce the exact black-view contract.
            left_wrist = np.zeros(EGO_VLA_RAW_IMAGE_SHAPE, dtype=np.uint8)
            right_wrist = np.zeros(EGO_VLA_RAW_IMAGE_SHAPE, dtype=np.uint8)
        images = {
            "front": front,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
        }
    else:
        images = {
            video_key: _to_rgb_hwc(_extract_image(obs, candidates))
            for video_key, candidates in VIDEO_KEY_CANDIDATES.items()
        }
    state_groups = _pack_state_groups(obs, action_type, robot_action_dim_info)

    return {
        "video": {
            key: np.asarray(image, dtype=np.uint8)[None, None, ...]
            for key, image in images.items()
        },
        "state": {
            key: value[None, None, :] for key, value in state_groups.items()
        },
        "language": {
            "annotation.human.task_description": [[prompt]],
        },
    }


def _gr00t_action_to_env(
    action: dict[str, np.ndarray],
    action_type: str,
    robot_action_dim_info: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    if action_type != "joint":
        raise ValueError(
            "GR00T_N17 is configured for joint-space relative actions "
            f"(action_type=joint), got {action_type!r}."
        )

    groups_by_name: dict[str, np.ndarray] = {}
    horizon: int | None = None
    for group_name, group_dim in _gr00t_group_dims(robot_action_dim_info):
        if group_name not in action:
            raise KeyError(f"GR00T action is missing modality group {group_name!r}.")
        group = np.asarray(action[group_name][0], dtype=np.float32)
        if group.ndim != 2 or group.shape[1] != group_dim:
            raise ValueError(
                f"GR00T action group {group_name!r} must have shape [T, {group_dim}], "
                f"got {group.shape}."
            )
        if horizon is None:
            horizon = group.shape[0]
        elif group.shape[0] != horizon:
            raise ValueError(
                f"GR00T action horizon mismatch for {group_name!r}: "
                f"expected {horizon}, got {group.shape[0]}."
            )
        groups_by_name[group_name] = group

    packed_action = np.concatenate(
        [groups_by_name[name] for name, _ in _physical_group_dims(robot_action_dim_info)],
        axis=-1,
    )
    unpacked = unpack_robot_state(
        packed_action,
        action_type=action_type,
        robot_action_dim_info=robot_action_dim_info,
        source_type="obs",
    )
    return [
        {key: np.asarray(value, dtype=np.float32) for key, value in step.items()}
        for step in unpacked
    ]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.model_cfg = model_cfg
        self.action_type = model_cfg.get("action_type", "joint")
        self.default_prompt = model_cfg.get(
            "default_prompt",
            model_cfg.get("task_name", "Perform the robot manipulation task."),
        )
        self.env_cfg_type = model_cfg["env_cfg_type"]
        self.robot_action_dim_info = _resolve_robot_action_dim_info(self.env_cfg_type)
        self.device = model_cfg.get("device", "cuda:0" if self._has_cuda() else "cpu")

        _load_modality_config(self.env_cfg_type)
        checkpoint_dir = _resolve_checkpoint_dir(model_cfg)
        embodiment_tag = model_cfg.get("embodiment_tag", "NEW_EMBODIMENT")
        cosmos_model = _resolve_cosmos_model(model_cfg)

        with network_checkpoint_view(checkpoint_dir) as network_dir, _cpu_checkpoint_view(network_dir, self.device) as load_dir:
            with _override_processor_cosmos_model(load_dir, cosmos_model):
                self.policy = Gr00tPolicy(
                    model_path=str(load_dir),
                    embodiment_tag=embodiment_tag,
                    device=self.device,
                    strict=True,
                )
        self.model = self.policy
        expected_groups = [name for name, _ in _gr00t_group_dims(self.robot_action_dim_info)]
        for modality in ("state", "action"):
            actual_groups = list(self.policy.modality_configs[modality].modality_keys)
            if actual_groups != expected_groups:
                raise ValueError(
                    f"Checkpoint {modality} modalities {actual_groups} do not match "
                    f"env_cfg_type={self.env_cfg_type!r} metadata groups {expected_groups}. "
                    "Use a checkpoint trained for this robot schema."
                )
        self.action_horizon = len(self.policy.modality_configs["action"].delta_indices)

        self._obs_list: list[dict[str, Any]] = []
        self._latest_env_idx_list: list[int] = [0]

        print(f"[GR00T_N17] Loaded checkpoint from {checkpoint_dir}")
        print(f"[GR00T_N17] cosmos_model={cosmos_model}")
        print(f"[GR00T_N17] action_groups={expected_groups}")
        print(f"[GR00T_N17] action_horizon={self.action_horizon}, embodiment_tag={embodiment_tag}")

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [
            obs.get("env_idx", index) for index, obs in enumerate(obs_list)
        ]
        self._obs_list = [
            _encode_observation(
                obs,
                self.default_prompt,
                self.action_type,
                self.robot_action_dim_info,
                self.env_cfg_type,
            )
            for obs in obs_list
        ]

    def get_action(self, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if not self._obs_list:
            raise AssertionError("update_obs or update_obs_batch first!")

        action_list = []
        for encoded_obs in self._obs_list:
            gr00t_action, _ = self.policy.get_action(encoded_obs, **kwargs)
            action_list.append(
                _gr00t_action_to_env(
                    gr00t_action,
                    self.action_type,
                    self.robot_action_dim_info,
                )
            )
        return action_list

    def reset(self):
        self._obs_list = []
        self._latest_env_idx_list = [0]
        self.policy.reset()
