"""ACT policy adapter and lazy deployment runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from revo_policy_deploy.deploy.policy_contract import load_contract
from revo_policy_deploy.policy.adapter import NamedObservationAdapter
from revo_policy_deploy.policy.loader import LoadedPolicy, load_policy


class ACTPolicy:
    """Run a loaded ACT model with contract-defined observation semantics."""

    def __init__(self, loaded: LoadedPolicy):
        """Create a stateful ACT adapter from a loaded policy."""
        self.loaded = loaded
        self.adapter = NamedObservationAdapter(loaded.contract)
        self.previous_observation: dict[str, Any] | None = None
        self.task_id: int | None = None

    def reset(self) -> None:
        """Clear model queues, temporal history, and task state."""
        self.loaded.policy.reset()
        self.previous_observation = None
        self.task_id = None

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run one named observation and return grouped absolute action targets."""
        import torch

        current = self.adapter.build_batch(observation)
        task_value = current.get("task_uid")
        current_task = int(task_value.reshape(-1)[0].item()) if task_value is not None else None
        if current_task != self.task_id:
            self.previous_observation = None
            self.task_id = current_task
        n_obs_steps = int(getattr(self.loaded.config, "n_obs_steps", 1))
        if n_obs_steps not in (1, 2):
            raise ValueError(f"unsupported ACT observation history length: {n_obs_steps}")
        if n_obs_steps == 2:
            previous = self.previous_observation or current
            batch = self.adapter.add_temporal_batch_dim(self.adapter.stack_history(previous, current))
        else:
            batch = current
        self.previous_observation = self.adapter.copy_observation(current)
        processed = self.loaded.preprocessor(batch)
        with torch.inference_mode():
            output = self.loaded.policy.predict_action_chunk(processed)
        output = self.loaded.postprocessor(output)
        actions = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        chunk_steps = int(self.loaded.contract["chunk_steps"])
        if actions.ndim != 2 or chunk_steps > actions.shape[0]:
            raise ValueError(f"policy output is incompatible with contract: shape={actions.shape}")
        return self.adapter.group_actions(actions[:chunk_steps])


class PolicyRuntime:
    """Lazy artifact runtime used by the protocol process and direct tests."""

    def __init__(self, package_dir: str | Path, *, device: str | None = None):
        """Read only the lightweight contract; defer model imports until LOAD."""
        self.package_dir = Path(package_dir).expanduser().resolve()
        self.contract = load_contract(self.package_dir)
        self.device = device or str(self.contract.get("runtime", {}).get("device", "auto"))
        self.policy: ACTPolicy | None = None

    @property
    def loaded(self) -> bool:
        """Return whether the heavyweight model has been loaded."""
        return self.policy is not None

    def load(self) -> None:
        """Validate and load the artifact exactly once."""
        if self.policy is None:
            self.policy = ACTPolicy(load_policy(self.package_dir, device=self.device))

    def predict(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        """Predict one action chunk after an explicit LOAD."""
        if self.policy is None:
            raise RuntimeError("inferencer is not loaded; send LOAD first")
        return self.policy.predict(observation)

    def reset(self) -> None:
        """Clear policy queues and episode-local state when loaded."""
        if self.policy is not None:
            self.policy.reset()
