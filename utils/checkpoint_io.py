"""Materialize network checkpoint shards before safetensors maps their pages."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def _create_memfd(name: str) -> int:
    # Portable Python builds can omit the wrapper despite host Linux support.
    # These flags and seals are Linux UAPI constants, independent of Python ABI.
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, 1 | 2)  # MFD_CLOEXEC | MFD_ALLOW_SEALING
    import ctypes
    create = ctypes.CDLL(None, use_errno=True).memfd_create
    create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    create.restype = ctypes.c_int
    fd = create(name.encode(), 1 | 2)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def _mount_type(path: Path) -> str:
    """Use the longest matching mount; mountinfo escapes spaces as octal."""
    target = str(path.resolve())
    best = (0, "")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        mount = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), left.split()[4])
        if (target == mount or target.startswith(mount.rstrip("/") + "/")) and len(mount) > best[0]:
            best = (len(mount), right.split()[0])
    return best[1]


def _signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _memory_available() -> int:
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    for limit_name, used_name in [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"), ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
        try:
            available = min(available, max(0, int(Path(limit_name).read_text()) - int(Path(used_name).read_text())))
        except (OSError, ValueError):
            pass
    return available


def _weight_files(checkpoint: Path) -> list[Path]:
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (checkpoint / "model.safetensors").is_file():
        names = ["model.safetensors"]
    else:
        return []
    if not names or any(not isinstance(n, str) or Path(n).name != n or not n.endswith(".safetensors") for n in names):
        raise ValueError("Checkpoint shard index must contain local safetensors filenames")
    return [checkpoint / name for name in names]


@contextmanager
def network_checkpoint_view(checkpoint: Path):
    """Keep exact shard bytes in sealed RAM files during native model loading.

    Local filesystems use the original path. This is not a page-cache prefetch:
    safetensors maps the sealed memfd files, never the source network weights.
    Descriptors and the private metadata view are released on success or error.
    The caller must finish model/processor construction inside this context.
    """
    checkpoint = Path(checkpoint)
    if sys.platform != "linux" or _mount_type(checkpoint) not in {"fuse.juicefs", "nfs", "nfs4", "fuse.sshfs"}:
        yield checkpoint
        return
    weights = _weight_files(checkpoint)
    if not weights:
        yield checkpoint
        return
    required = 3 * sum(p.stat().st_size for p in weights) + 4 * 1024**3
    if _memory_available() < required:
        print(f"[checkpoint-io] insufficient staging headroom; using original loader source={checkpoint}", flush=True)
        yield checkpoint
        return
    import fcntl

    descriptors = []
    records = []
    started = time.monotonic()
    print(f"[checkpoint-io] staging {len(weights)} network shards source={checkpoint}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="xpolicy-checkpoint-view-") as directory:
            view = Path(directory)
            index = checkpoint / "model.safetensors.index.json"
            index_signature = _signature(index.stat()) if index.exists() else None
            for source in weights:
                before = source.stat()
                fd = _create_memfd("xpolicy-" + source.name)
                descriptors.append(fd)
                digest = hashlib.sha256()
                copied = 0
                last_report = time.monotonic()
                with source.open("rb", buffering=0) as reader, os.fdopen(os.dup(fd), "wb", buffering=0) as writer:
                    if _signature(os.fstat(reader.fileno())) != _signature(before):
                        raise RuntimeError(f"Checkpoint changed before reading: {source}")
                    while block := reader.read(4 * 1024 * 1024):
                        pending = memoryview(block)
                        while pending:
                            written = writer.write(pending)
                            if not written:
                                raise OSError("Checkpoint RAM file write made no progress")
                            pending = pending[written:]
                        digest.update(block)
                        copied += len(block)
                        if time.monotonic() - last_report >= 15:
                            print(f"[checkpoint-io] read {source.name} bytes={copied}/{before.st_size}", flush=True)
                            last_report = time.monotonic()
                    if _signature(os.fstat(reader.fileno())) != _signature(before):
                        raise RuntimeError(f"Checkpoint changed while reading: {source}")
                if copied != before.st_size or _signature(source.stat()) != _signature(before):
                    raise RuntimeError(f"Checkpoint changed while staging: {source}")
                fcntl.fcntl(fd, 1033, 8 | 4 | 2 | 1)  # F_ADD_SEALS: WRITE|GROW|SHRINK|SEAL
                (view / source.name).symlink_to(f"/proc/{os.getpid()}/fd/{fd}")
                records.append({"name": source.name, "bytes": copied, "sha256": digest.hexdigest()})
            weight_names = {p.name for p in weights}
            for entry in checkpoint.iterdir():
                if entry.name in weight_names:
                    continue
                destination = view / entry.name
                if entry.is_file() and entry.stat().st_size <= 16 * 1024 * 1024:
                    # The processor may override its local model path. Keep the
                    # source metadata immutable even during concurrent loads.
                    shutil.copy2(entry, destination)
                else:
                    destination.symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
            if (_signature(index.stat()) if index.exists() else None) != index_signature:
                raise RuntimeError(f"Checkpoint shard index changed while staging: {index}")
            print("[checkpoint-io] ready " + json.dumps({"source": str(checkpoint), "storage": "sealed-memfd", "seconds": round(time.monotonic() - started, 3), "shards": records}), flush=True)
            yield view
    finally:
        for fd in descriptors:
            os.close(fd)
