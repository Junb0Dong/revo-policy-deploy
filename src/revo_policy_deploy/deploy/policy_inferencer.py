"""Framed-stdio lifecycle loop for a packaged Revo ACT policy."""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
import traceback
from pathlib import Path

from revo_policy_deploy.deploy.inferencer_protocol import recv_frame, send_frame
from revo_policy_deploy.policy.policy import PolicyRuntime


def serve(
    package_dir: str | Path, *, protocol_in=None, protocol_out=None, device: str | None = None
) -> None:
    """Serve one policy artifact over the framed deployment protocol."""
    protocol_in = sys.stdin.buffer if protocol_in is None else protocol_in
    protocol_out = sys.stdout.buffer if protocol_out is None else protocol_out
    runtime = PolicyRuntime(package_dir, device=device)
    while True:
        message = recv_frame(protocol_in)
        message_type = str(message.get("type", "")).upper()
        try:
            if message_type == "DESCRIBE":
                send_frame(
                    protocol_out,
                    {
                        "type": "CONTRACT",
                        "contract": runtime.contract,
                        "capabilities": {
                            "inference_strategies": runtime.contract.get(
                                "inference_strategies", ["standard"]
                            )
                        },
                    },
                )
            elif message_type == "LOAD":
                _validate_load_request(message, runtime.contract)
                runtime.load()
                send_frame(
                    protocol_out,
                    {
                        "type": "READY",
                        "checkpoint_id": runtime.contract["checkpoint_id"],
                        "contract_hash": runtime.contract["contract_hash"],
                        "inference_strategy": "standard",
                    },
                )
            elif message_type == "INFER":
                started = time.perf_counter()
                grouped = runtime.predict(message["observation"])
                send_frame(
                    protocol_out,
                    {
                        "type": "RESULT",
                        "request_id": int(message["request_id"]),
                        "contract_hash": runtime.contract["contract_hash"],
                        "actions": grouped,
                        "timing": {"inferencer_total_ms": (time.perf_counter() - started) * 1000.0},
                    },
                )
            elif message_type == "RESET":
                runtime.reset()
                send_frame(protocol_out, {"type": "RESET_DONE"})
            elif message_type == "CLOSE":
                send_frame(protocol_out, {"type": "CLOSED"})
                return
            else:
                raise ValueError(f"unsupported message type: {message_type!r}")
        except Exception as exc:
            send_frame(
                protocol_out,
                {
                    "type": "ERROR",
                    "request_id": message.get("request_id"),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def _validate_load_request(message: dict, contract: dict) -> None:
    expected_id = message.get("checkpoint_id")
    expected_hash = message.get("contract_hash")
    if expected_id is not None and str(expected_id) != str(contract["checkpoint_id"]):
        raise ValueError("LOAD checkpoint_id does not match policy contract")
    if expected_hash is not None and str(expected_hash) != str(contract["contract_hash"]):
        raise ValueError("LOAD contract_hash does not match policy contract")


def main() -> None:
    """Run the inferencer command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    with contextlib.suppress(KeyboardInterrupt, EOFError):
        serve(args.checkpoint, protocol_out=protocol_out, device=args.device)


if __name__ == "__main__":
    main()
