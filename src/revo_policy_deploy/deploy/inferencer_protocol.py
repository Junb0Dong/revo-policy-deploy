"""Length-prefixed JSON frames with optional raw NumPy array blobs."""

from __future__ import annotations

import json
import struct
from typing import Any, BinaryIO

import numpy as np

_HEADER = struct.Struct("!Q")
_ARRAY_MARKER = "__ndarray_blob__"
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MAX_BLOB_BYTES = 1024 * 1024 * 1024


def send_frame(stream: BinaryIO, payload: dict[str, Any]) -> None:
    """Encode and flush one protocol frame."""
    blobs: list[bytes] = []
    encoded = _encode(payload, blobs)
    header = json.dumps(
        {"payload": encoded, "blob_sizes": [len(blob) for blob in blobs]},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    stream.write(_HEADER.pack(len(header)))
    stream.write(header)
    stream.writelines(blobs)
    stream.flush()


def recv_frame(stream: BinaryIO) -> dict[str, Any]:
    """Read, validate, and decode one protocol frame."""
    header_size = _HEADER.unpack(_read_exact(stream, _HEADER.size))[0]
    if header_size > _MAX_HEADER_BYTES:
        raise ValueError("invalid frame header size; stdout may contain non-protocol output")
    header = json.loads(_read_exact(stream, header_size).decode())
    blob_sizes = [int(size) for size in header.get("blob_sizes", [])]
    if any(size < 0 for size in blob_sizes) or sum(blob_sizes) > _MAX_BLOB_BYTES:
        raise ValueError("frame blobs exceed the protocol safety limit")
    blobs = [_read_exact(stream, size) for size in blob_sizes]
    payload = _decode(header["payload"], blobs)
    if not isinstance(payload, dict):
        raise ValueError("protocol payload must be an object")
    return payload


def _encode(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.kind not in "biufc":
            raise ValueError("protocol ndarray dtype must be numeric")
        index = len(blobs)
        blobs.append(array.tobytes(order="C"))
        return {_ARRAY_MARKER: index, "dtype": array.dtype.str, "shape": list(array.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _encode(item, blobs) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item, blobs) for item in value]
    return value


def _decode(value: Any, blobs: list[bytes]) -> Any:
    if isinstance(value, dict) and _ARRAY_MARKER in value:
        index = int(value[_ARRAY_MARKER])
        if index < 0 or index >= len(blobs):
            raise ValueError(f"invalid ndarray blob index: {index}")
        try:
            dtype = np.dtype(value["dtype"])
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid ndarray dtype") from exc
        if dtype.kind not in "biufc":
            raise ValueError("protocol ndarray dtype must be numeric")
        raw_shape = value.get("shape")
        if not isinstance(raw_shape, list) or len(raw_shape) > 16:
            raise ValueError("invalid ndarray shape")
        shape = tuple(int(dimension) for dimension in raw_shape)
        if any(dimension < 0 for dimension in shape):
            raise ValueError("invalid ndarray shape")
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if expected != len(blobs[index]):
            raise ValueError("ndarray blob size does not match dtype and shape")
        return np.frombuffer(blobs[index], dtype=dtype).reshape(shape)
    if isinstance(value, dict):
        return {key: _decode(item, blobs) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item, blobs) for item in value]
    return value


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("protocol stream closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
