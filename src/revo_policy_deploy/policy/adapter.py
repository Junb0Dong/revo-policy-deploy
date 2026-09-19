"""Named observation/action adapter for the Revo policy contract."""

from __future__ import annotations

from typing import Any

import numpy as np


class NamedObservationAdapter:
    """Convert protocol observations to model tensors and group model actions."""

    def __init__(self, contract: dict[str, Any]):
        """Bind the adapter to one immutable contract."""
        self.contract = contract

    def resolve_task_id(self, task: Any = None, task_index: Any = None) -> int | None:
        """Resolve task text or an explicit task index using the frozen task map."""
        task_map = self.contract.get("task_map")
        num_tasks = int(self.contract.get("num_tasks", 1) or 1)
        if num_tasks == 1:
            if task_index is not None and int(task_index) != 0:
                raise ValueError(f"unknown task_index {int(task_index)} for single-task policy")
            return 0
        if not isinstance(task_map, dict) or len(task_map) < num_tasks:
            raise ValueError("task map is required for a multi-task policy")
        values = {int(value) for value in task_map.values()}
        if task_index is not None:
            value = int(task_index)
            if value not in values:
                raise ValueError(f"unknown task_index {value}")
            return value
        if task is None:
            raise ValueError("observation task is required for a multi-task policy")
        key = str(task)
        if key not in task_map:
            raise ValueError(f"unknown task {key!r}")
        return int(task_map[key])

    def build_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Validate a raw named observation and create LeRobot input tensors."""
        import torch
        from torch.nn import functional as functional

        if not isinstance(observation, dict):
            raise ValueError("observation must be an object")
        received_groups = observation.get("joint_groups")
        if not isinstance(received_groups, dict):
            raise ValueError("observation joint_groups must be an object")
        state_parts: list[np.ndarray] = []
        for group, expected_names in self.contract["required_joint_groups"].items():
            received = received_groups.get(group)
            if not isinstance(received, dict):
                raise ValueError(f"observation missing joint group {group!r}")
            names = [str(name) for name in received.get("names", [])]
            positions = np.asarray(received.get("positions"), dtype=np.float32)
            if len(names) != len(set(names)):
                raise ValueError(f"joint group {group!r} contains duplicate names")
            if positions.shape != (len(names),):
                raise ValueError(f"joint group {group!r} names/positions mismatch")
            if not np.all(np.isfinite(positions)):
                raise ValueError(f"joint group {group!r} contains NaN or Inf")
            indices = {name: index for index, name in enumerate(names)}
            missing = [name for name in expected_names if name not in indices]
            if missing:
                raise ValueError(f"joint group {group!r} missing names: {missing}")
            state_parts.append(np.asarray([positions[indices[name]] for name in expected_names]))

        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(np.concatenate(state_parts)).to(dtype=torch.float32)
        }
        task_id = self.resolve_task_id(observation.get("task"), observation.get("task_index"))
        if task_id is not None:
            batch["task_uid"] = torch.tensor([task_id], dtype=torch.long)

        images = observation.get("images")
        if not isinstance(images, dict):
            raise ValueError("observation images must be an object")
        image_contract = self.contract["image_contract"]
        resize_hw = tuple(int(value) for value in image_contract["resize_hw"])
        native_hw = image_contract["native_hw"]
        for camera in self.contract["required_cameras"]:
            if camera not in images:
                raise ValueError(f"observation missing camera {camera!r}")
            image = np.asarray(images[camera])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"camera {camera!r} must be RGB uint8 HWC, got {image.shape}/{image.dtype}")
            expected_hw = tuple(int(value) for value in native_hw[camera])
            if tuple(image.shape[:2]) != expected_hw:
                raise ValueError(
                    f"camera {camera!r} has shape {tuple(image.shape[:2])}, expected {expected_hw}"
                )
            tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div(255.0)
            tensor = functional.interpolate(
                tensor.unsqueeze(0), size=resize_hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            batch[camera] = tensor.contiguous()
        return batch

    @staticmethod
    def copy_observation(batch: dict[str, Any]) -> dict[str, Any]:
        """Clone tensor observations so history cannot be mutated in place."""
        import torch

        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    @staticmethod
    def stack_history(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
        """Stack two observations in oldest-to-newest order."""
        import torch

        if previous.keys() != current.keys():
            raise ValueError("history observations must contain identical feature keys")
        result: dict[str, Any] = {}
        for key in current:
            old, new = previous[key], current[key]
            if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
                if key in {"task_uid", "task_index"}:
                    if not torch.equal(old, new):
                        raise ValueError(f"history metadata changed for {key!r}")
                    result[key] = new
                else:
                    result[key] = torch.stack((old, new), dim=0)
            elif old != new:
                raise ValueError(f"history metadata changed for {key!r}")
            else:
                result[key] = new
        return result

    @staticmethod
    def add_temporal_batch_dim(batch: dict[str, Any]) -> dict[str, Any]:
        """Promote a temporal observation to the model's batch dimension."""
        import torch

        return {
            key: value.unsqueeze(0)
            if isinstance(value, torch.Tensor) and key not in {"task_uid", "task_index"}
            else value
            for key, value in batch.items()
        }

    def group_actions(self, actions: np.ndarray) -> dict[str, np.ndarray]:
        """Validate and split a model action matrix in contract order."""
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or not np.all(np.isfinite(actions)):
            raise ValueError(f"policy output must be finite [steps, dim], got {actions.shape}")
        expected_dim = sum(len(names) for names in self.contract["output_joint_groups"].values())
        if actions.shape[1] != expected_dim:
            raise ValueError(f"policy output dim {actions.shape[1]} != contract dim {expected_dim}")
        grouped: dict[str, np.ndarray] = {}
        offset = 0
        for group, names in self.contract["output_joint_groups"].items():
            width = len(names)
            grouped[group] = np.ascontiguousarray(actions[:, offset : offset + width], dtype=np.float32)
            offset += width
        return grouped
