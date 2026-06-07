"""
Load PyTorch ``torch.save`` zip checkpoints as numpy arrays — no ``torch`` import.

Supports plain state-dict archives (e.g. DiffRhythm ``decoder.bin``, MuQ
``pytorch_model.bin``). Fails loud on pickled full ``nn.Module`` graphs.
"""
from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_DTYPE_BY_STORAGE: dict[str, np.dtype] = {
    "FloatStorage": np.dtype(np.float32),
    "HalfStorage": np.dtype(np.float16),
    "DoubleStorage": np.dtype(np.float64),
    "LongStorage": np.dtype(np.int64),
    "IntStorage": np.dtype(np.int32),
    "ShortStorage": np.dtype(np.int16),
    "CharStorage": np.dtype(np.int8),
    "ByteStorage": np.dtype(np.uint8),
    "BoolStorage": np.dtype(np.bool_),
}


class _NumpyStorage:
    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr


def _storage_dtype(storage_type: Any) -> np.dtype:
    name = getattr(storage_type, "__name__", str(storage_type))
    for key, dtype in _DTYPE_BY_STORAGE.items():
        if key in name:
            return dtype
    raise RuntimeError(f"Unsupported PyTorch storage type: {storage_type!r}")


def _rebuild_tensor_v2(
    storage: _NumpyStorage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: Any,
    metadata: Any = None,
) -> np.ndarray:
    del stride, requires_grad, backward_hooks, metadata
    arr = storage._arr
    offset = int(storage_offset or 0)
    if offset:
        arr = arr[offset:]
    numel = 1
    shape = tuple(int(x) for x in size)
    for dim in shape:
        numel *= dim
    return np.asarray(arr[:numel], dtype=arr.dtype).reshape(shape)


def _rebuild_tensor(
    storage: _NumpyStorage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
) -> np.ndarray:
    return _rebuild_tensor_v2(storage, storage_offset, size, stride, False, None)


class _TorchBinUnpickler(pickle.Unpickler):
    def __init__(self, file_obj: Any, *, storages: dict[str, _NumpyStorage], read_blob):
        super().__init__(file_obj)
        self._storages = storages
        self._read_blob = read_blob

    def persistent_load(self, pid: Any) -> _NumpyStorage:
        if not isinstance(pid, tuple) or len(pid) < 2 or pid[0] != "storage":
            raise RuntimeError(f"Unsupported PyTorch persistent id: {pid!r}")
        storage_type, key, _location, numel = pid[1], pid[2], pid[3], pid[4]
        key = str(key)
        if key in self._storages:
            return self._storages[key]
        dtype = _storage_dtype(storage_type)
        raw = self._read_blob(key)
        arr = np.frombuffer(raw, dtype=dtype, count=int(numel)).copy()
        if arr.size != int(numel):
            raise RuntimeError(
                f"PyTorch storage {key!r} expected {numel} elements, got {arr.size}"
            )
        storage = _NumpyStorage(arr)
        self._storages[key] = storage
        return storage

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_tensor":
                return _rebuild_tensor
        if module == "torch" and name == "Size":
            return tuple
        if module == "torch" and name.endswith("Storage"):
            return type(name, (), {"__module__": module})
        if module == "collections" and name == "OrderedDict":
            import collections

            return collections.OrderedDict
        if module.startswith("torch"):
            raise pickle.UnpicklingError(
                f"torch-free checkpoint loader cannot unpickle {module}.{name}; "
                "expected a flat state_dict archive"
            )
        return super().find_class(module, name)


def load_pytorch_bin(path: str | Path) -> Any:
    """Load a ``torch.save`` zip archive; tensor values become ``numpy.ndarray``."""
    archive = Path(path)
    if not archive.is_file():
        raise RuntimeError(f"PyTorch checkpoint not found: {archive}")

    with zipfile.ZipFile(archive, "r") as zf:
        pkl_names = [n for n in zf.namelist() if n.endswith("data.pkl")]
        if not pkl_names:
            raise RuntimeError(f"No data.pkl entry in PyTorch archive: {archive}")
        pkl_name = pkl_names[0]
        prefix = pkl_name.rsplit("data.pkl", 1)[0]
        storages: dict[str, _NumpyStorage] = {}

        def read_blob(key: str) -> bytes:
            return zf.read(f"{prefix}data/{key}")

        with zf.open(pkl_name, "r") as handle:
            unpickler = _TorchBinUnpickler(handle, storages=storages, read_blob=read_blob)
            return unpickler.load()


def state_dict_to_numpy(path: str | Path) -> dict[str, np.ndarray]:
    """Return a flat ``str -> ndarray`` map from a checkpoint file."""
    obj = load_pytorch_bin(path)
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"PyTorch checkpoint at {path} must deserialize to dict, got {type(obj)!r}"
        )
    out: dict[str, np.ndarray] = {}
    for key, val in obj.items():
        if isinstance(val, np.ndarray):
            out[str(key)] = val
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if not isinstance(sub_val, np.ndarray):
                    raise RuntimeError(
                        f"Nested checkpoint value {key}.{sub_key} is {type(sub_val)!r}, expected ndarray"
                    )
                out[f"{key}.{sub_key}"] = sub_val
        else:
            raise RuntimeError(f"Checkpoint entry {key!r} is {type(val)!r}, expected ndarray")
    return out
