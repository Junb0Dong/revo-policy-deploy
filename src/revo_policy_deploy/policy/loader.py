"""The only module that imports and loads the pinned LeRobot policy runtime."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from revo_policy_deploy.deploy.policy_contract import POLICY_LOADER_VERSION, load_contract


@dataclass
class LoadedPolicy:
    """Loaded model, processors, configuration, and artifact metadata."""

    policy: Any
    preprocessor: Any
    postprocessor: Any
    config: Any
    model_dir: Path
    contract: dict[str, Any]


def load_train_config(package_dir: str | Path, *, checkpoint_id: str) -> dict[str, Any]:
    """Validate the generated artifact header and every referenced runtime path."""
    package = Path(package_dir).expanduser().resolve()
    path = package / "train_config.yaml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read train_config.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("train_config.yaml must contain an object")
    for field in ("generated", "do_not_edit"):
        if payload.get(field) is not True:
            raise ValueError(f"train_config.yaml requires {field}: true")
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise ValueError("train_config.yaml requires schema_version: 1")
    if not str(payload.get("generated_by", "")).strip():
        raise ValueError("train_config.yaml requires generated_by")
    if str(payload.get("checkpoint_id", "")) != checkpoint_id:
        raise ValueError("train_config.yaml checkpoint_id differs from policy_contract.json")
    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("train_config.yaml requires a runtime object")
    for field in ("weights", "model_config", "model_dir"):
        value = runtime.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"train_config.yaml runtime requires {field}")
        candidate = (package / value).resolve()
        if not candidate.is_relative_to(package):
            raise ValueError(f"train_config.yaml runtime.{field} escapes checkpoint")
        if not candidate.exists():
            raise FileNotFoundError(f"train_config.yaml runtime.{field} is missing: {value}")
    expected_hash = payload.get("content_sha256")
    if expected_hash:
        digest = hashlib.sha256()
        for field in ("weights", "model_config"):
            relative = str(runtime[field])
            digest.update(relative.encode())
            digest.update((package / relative).read_bytes())
        if digest.hexdigest() != str(expected_hash):
            raise ValueError("train_config.yaml content_sha256 mismatch")
    return payload


def load_policy(package_dir: str | Path, *, device: str = "auto") -> LoadedPolicy:
    """Load an ACT model and saved processors using only artifact-local paths."""
    package = Path(package_dir).expanduser().resolve()
    contract = load_contract(package)
    _validate_runtime_compatibility(contract)
    train_config = load_train_config(package, checkpoint_id=str(contract["checkpoint_id"]))
    model_dir = (package / str(train_config["runtime"]["model_dir"])).resolve()

    import torch
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class, make_pre_post_processors

    resolved_device = _resolve_device(device, torch)
    config = PreTrainedConfig.from_pretrained(model_dir)
    if str(config.type) != "act":
        raise ValueError(f"only ACT artifacts are supported, got policy type {config.type!r}")
    expected_family = contract.get("policy_family")
    if expected_family and str(config.type) != str(expected_family):
        raise ValueError(f"policy family mismatch: contract={expected_family!r}, config={config.type!r}")
    config.device = resolved_device
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(model_dir, config=config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": resolved_device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return LoadedPolicy(policy, preprocessor, postprocessor, config, model_dir, contract)


def _validate_runtime_compatibility(contract: dict[str, Any]) -> None:
    runtime = contract.get("runtime", {})
    compatibility = runtime.get("compatibility", {}) if isinstance(runtime, dict) else {}
    loader = compatibility.get("policy_loader_version")
    if loader is not None and str(loader) != POLICY_LOADER_VERSION:
        raise ValueError(
            f"artifact requires policy loader {loader}, current loader is {POLICY_LOADER_VERSION}"
        )
    requested_lerobot = compatibility.get("lerobot_version")
    installed_lerobot = version("lerobot")
    if requested_lerobot not in (None, "", "unknown") and str(requested_lerobot) != installed_lerobot:
        raise ValueError(f"artifact requires LeRobot {requested_lerobot}, installed {installed_lerobot}")


def _resolve_device(device: str | None, torch: Any) -> str:
    if device in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)
