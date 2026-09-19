import io
import json
from pathlib import Path

import numpy as np
import pytest

from revo_policy_deploy.deploy.inferencer_protocol import recv_frame, send_frame
from revo_policy_deploy.deploy.policy_contract import load_contract, payload_hash
from revo_policy_deploy.deploy.policy_inferencer import serve
from revo_policy_deploy.policy.adapter import NamedObservationAdapter
from revo_policy_deploy.policy.loader import load_train_config


def make_contract() -> dict:
    value = {
        "schema_version": 1,
        "generated": True,
        "do_not_edit": True,
        "generated_by": "test",
        "protocol_version": 1,
        "checkpoint_id": "test:1",
        "required_joint_groups": {"left_arm": ["la0", "la1"]},
        "required_cameras": ["cam"],
        "output_joint_groups": {"left_arm": ["la0", "la1"]},
        "state_dim": 2,
        "action_dim": 2,
        "chunk_steps": 2,
        "model_horizon": 2,
        "policy_rate_hz": 30.0,
        "num_tasks": 1,
        "task_map": {},
        "action_semantics": "absolute_joint_position",
        "image_contract": {"native_hw": {"cam": [2, 3]}, "resize_hw": [4, 4]},
        "runtime": {"device": "cpu"},
    }
    value["contract_hash"] = payload_hash(value)
    return value


def write_contract(path: Path, contract: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "policy_contract.json").write_text(json.dumps(contract), encoding="utf-8")


def test_named_observation_reorders_groups_and_images() -> None:
    adapter = NamedObservationAdapter(make_contract())
    batch = adapter.build_batch(
        {
            "joint_groups": {"left_arm": {"names": ["la1", "la0"], "positions": [2.0, 1.0]}},
            "images": {"cam": np.zeros((2, 3, 3), dtype=np.uint8)},
        }
    )
    np.testing.assert_array_equal(batch["observation.state"].numpy(), [1.0, 2.0])
    assert tuple(batch["cam"].shape) == (3, 4, 4)


def test_named_observation_rejects_bad_inputs() -> None:
    adapter = NamedObservationAdapter(make_contract())
    base = {
        "joint_groups": {"left_arm": {"names": ["la0", "la1"], "positions": [0.0, 1.0]}},
        "images": {"cam": np.zeros((2, 3, 3), dtype=np.uint8)},
    }
    base["joint_groups"]["left_arm"]["positions"][0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        adapter.build_batch(base)
    base["joint_groups"]["left_arm"]["positions"][0] = 0.0
    base["images"]["cam"] = np.zeros((3, 3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        adapter.build_batch(base)


def test_contract_hash_and_group_validation(tmp_path: Path) -> None:
    contract = make_contract()
    path = tmp_path / "checkpoint"
    write_contract(path, contract)
    assert load_contract(path)["checkpoint_id"] == "test:1"
    contract["output_joint_groups"] = {"wrist": ["la0"]}
    contract["contract_hash"] = payload_hash(contract)
    (path / "policy_contract.json").write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown joint group"):
        load_contract(path)


def test_train_config_rejects_path_escape(tmp_path: Path) -> None:
    package = tmp_path / "checkpoint"
    package.mkdir()
    (package / "train_config.yaml").write_text(
        "schema_version: 1\n"
        "generated: true\n"
        "do_not_edit: true\n"
        "generated_by: test\n"
        "checkpoint_id: test:1\n"
        "runtime:\n"
        "  weights: ../weights.safetensors\n"
        "  model_config: config.json\n"
        "  model_dir: .\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="escapes"):
        load_train_config(package, checkpoint_id="test:1")


def test_describe_does_not_load_model_and_close() -> None:
    incoming = io.BytesIO()
    outgoing = io.BytesIO()
    send_frame(incoming, {"type": "DESCRIBE"})
    send_frame(incoming, {"type": "CLOSE"})
    incoming.seek(0)
    # The contract lives in a temporary artifact so construction only reads JSON.
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        write_contract(Path(directory), make_contract())
        serve(directory, protocol_in=incoming, protocol_out=outgoing, device="cpu")
    outgoing.seek(0)
    assert recv_frame(outgoing)["type"] == "CONTRACT"
    assert recv_frame(outgoing)["type"] == "CLOSED"


def test_protocol_round_trip_and_rejects_object_dtype() -> None:
    stream = io.BytesIO()
    send_frame(stream, {"value": np.arange(4, dtype=np.float32)})
    stream.seek(0)
    np.testing.assert_array_equal(recv_frame(stream)["value"], np.arange(4, dtype=np.float32))
    stream = io.BytesIO()
    with pytest.raises(ValueError, match="numeric"):
        send_frame(stream, {"value": np.array([object()], dtype=object)})
