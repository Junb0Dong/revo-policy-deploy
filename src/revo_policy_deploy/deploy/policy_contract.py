"""Generated, immutable interface contract for a Revo deployment artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CONTRACT_FILENAME = "policy_contract.json"
SCHEMA_VERSION = 1
PROTOCOL_VERSION = 1
POLICY_LOADER_VERSION = "1"
JOINT_GROUP_ORDER = ("left_arm", "right_arm", "left_hand", "right_hand")


def payload_hash(payload: dict[str, Any]) -> str:
    """Hash a contract while excluding its self-referential hash field."""
    unsigned = {key: value for key, value in payload.items() if key != "contract_hash"}
    encoded = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_contract(path_or_dir: str | Path) -> dict[str, Any]:
    """Load and validate a generated contract without importing the model runtime."""
    path = Path(path_or_dir).expanduser()
    if path.is_dir():
        path = path / CONTRACT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing {CONTRACT_FILENAME}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON contract: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"contract must be a JSON object: {path}")
    if payload.get("generated") is not True or payload.get("do_not_edit") is not True:
        raise ValueError(f"contract must be generated and immutable: {path}")
    if int(payload.get("protocol_version", 0) or 0) != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {payload.get('protocol_version')}")
    if "schema_version" in payload and int(payload.get("schema_version", 0) or 0) != SCHEMA_VERSION:
        raise ValueError(f"unsupported contract schema version: {payload.get('schema_version')}")
    expected = str(payload.get("contract_hash", ""))
    if not expected or expected != payload_hash(payload):
        raise ValueError(f"contract hash mismatch: {path}")
    _validate_contract(payload, path)
    return payload


def _validate_contract(payload: dict[str, Any], path: Path) -> None:
    required = (
        "checkpoint_id",
        "required_joint_groups",
        "required_cameras",
        "output_joint_groups",
        "chunk_steps",
        "policy_rate_hz",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"contract is missing fields {missing}: {path}")
    if not isinstance(payload["checkpoint_id"], str) or not payload["checkpoint_id"].strip():
        raise ValueError(f"contract checkpoint_id must be a non-empty string: {path}")
    input_dim = _validate_groups(payload["required_joint_groups"], "required_joint_groups", path)
    output_dim = _validate_groups(payload["output_joint_groups"], "output_joint_groups", path)
    if payload.get("state_dim") is not None and int(payload["state_dim"]) != input_dim:
        raise ValueError(f"contract input groups do not cover state_dim: {path}")
    if payload.get("action_dim") is not None and int(payload["action_dim"]) != output_dim:
        raise ValueError(f"contract output groups do not cover action_dim: {path}")
    cameras = payload["required_cameras"]
    if not isinstance(cameras, list) or not cameras or len(cameras) != len(set(cameras)):
        raise ValueError(f"required_cameras must be a non-empty unique list: {path}")
    if int(payload["chunk_steps"]) <= 0 or float(payload["policy_rate_hz"]) <= 0:
        raise ValueError(f"chunk_steps and policy_rate_hz must be positive: {path}")
    horizon = payload.get("model_horizon")
    if horizon is not None and int(payload["chunk_steps"]) > int(horizon):
        raise ValueError(f"chunk_steps exceeds model_horizon: {path}")
    if int(payload.get("history_observation_stride", 0) or 0) < 0:
        raise ValueError(f"history_observation_stride must be non-negative: {path}")
    if payload.get("action_semantics") not in (None, "absolute_joint_position"):
        raise ValueError(f"unsupported action semantics: {payload.get('action_semantics')}")
    family = payload.get("policy_family")
    if family not in (None, "", "act"):
        raise ValueError(f"unsupported policy family: {family}")
    _validate_images(payload, path)


def _validate_groups(value: Any, field: str, path: Path) -> int:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"contract {field} must be a non-empty object: {path}")
    keys = list(value)
    if any(key not in JOINT_GROUP_ORDER for key in keys):
        raise ValueError(f"contract {field} contains an unknown joint group: {path}")
    if keys != [key for key in JOINT_GROUP_ORDER if key in value]:
        raise ValueError(f"contract {field} must use canonical group order: {path}")
    seen: set[str] = set()
    width = 0
    for group, raw_names in value.items():
        if not isinstance(raw_names, list) or not raw_names:
            raise ValueError(f"contract {field}.{group} must be a non-empty list: {path}")
        names = [str(name) for name in raw_names]
        if len(names) != len(set(names)):
            raise ValueError(f"contract {field}.{group} contains duplicate names: {path}")
        overlap = seen.intersection(names)
        if overlap:
            raise ValueError(f"contract {field} repeats names {sorted(overlap)}: {path}")
        seen.update(names)
        width += len(names)
    return width


def _validate_images(payload: dict[str, Any], path: Path) -> None:
    image_contract = payload.get("image_contract")
    if not isinstance(image_contract, dict):
        raise ValueError(f"contract image_contract must be an object: {path}")
    resize = image_contract.get("resize_hw")
    native = image_contract.get("native_hw")
    if not isinstance(resize, list) or len(resize) != 2 or any(int(value) <= 0 for value in resize):
        raise ValueError(f"contract image_contract.resize_hw is invalid: {path}")
    if not isinstance(native, dict) or set(native) != set(payload["required_cameras"]):
        raise ValueError(f"image_contract.native_hw must cover required cameras: {path}")
    for camera, shape in native.items():
        if not isinstance(shape, list) or len(shape) != 2 or any(int(value) <= 0 for value in shape):
            raise ValueError(f"native camera shape is invalid for {camera}: {path}")
