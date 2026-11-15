"""Two-phase checkpointing for PyTorch, plus a PyTorch Distributed Checkpoint baseline.

Phase one copies the model and optimizer state into page-locked CPU memory;
the training loop is blocked only for this copy (the stall). Phase two
writes the copy to storage on a background thread. A checkpoint counts only
once its write has finished and the manifest points to it. At most one
write is in flight: a save that finds the previous write still running
waits for it, and that wait is counted as stall.

Both checkpointers have the same interface, so the training loop needs two
calls: `restore()` once at start, and `after_step(step, seconds)` after
every step.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Callable

import torch

from facp.policies import Measured, Policy
from facp.record import Record

MANIFEST = "latest.json"


class _CrcWriter:
    """File wrapper that computes a CRC-32 of everything written through it."""

    def __init__(self, f):
        self.f = f
        self.crc = 0
        self.bytes = 0

    def write(self, data) -> int:
        view = memoryview(data)
        self.crc = zlib.crc32(view, self.crc)
        self.bytes += view.nbytes
        return self.f.write(data)

    def flush(self) -> None:
        self.f.flush()


def file_crc(path: Path, chunk: int = 64 << 20) -> int:
    crc = 0
    with open(path, "rb") as f:
        while block := f.read(chunk):
            crc = zlib.crc32(block, crc)
    return crc


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class _Base:
    def __init__(self, directory: str | os.PathLike, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer, policy: Policy, record: Record):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.policy = policy
        self.record = record
        self.measured = Measured()
        self.next_save = policy.first_steps
        self._writing = threading.Event()

    # -- interface used by the training loop ---------------------------------

    def after_step(self, step: int, seconds: float) -> None:
        """Report a finished step (1-based) and save if the policy says so."""
        writing = self.writing
        self.measured.update("step_writing" if writing else "step_clean", seconds)
        self.record.write("step_time", step=step, seconds=seconds, writing=writing)
        if step >= self.next_save:
            self.save(step)

    @property
    def writing(self) -> bool:
        return self._writing.is_set()

    def manifest(self) -> dict | None:
        path = self.dir / MANIFEST
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    # -- shared helpers -------------------------------------------------------

    def _after_write(self, step: int, began: float, info: dict) -> None:
        seconds = time.time() - began
        self.measured.update("write", seconds)
        self.record.write("write_end", step=step, write=seconds, **info)

    def _plan_next(self, step: int) -> None:
        k = self.policy.interval_steps(self.measured)
        self.next_save = step + k
        costs = self.measured.costs().__dict__ if self.measured.ready() else None
        self.record.write("interval", step=step, steps=k, policy=self.policy.name, costs=costs)


class TwoPhaseCheckpointer(_Base):
    def __init__(self, *args, keep: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep = keep
        self._buffers: dict[str, torch.Tensor] = {}
        self._thread: threading.Thread | None = None
        for stale in self.dir.glob("*.tmp"):
            stale.unlink()

    def _copy(self, obj: Any, path: str) -> Any:
        if isinstance(obj, torch.Tensor):
            buf = self._buffers.get(path)
            if buf is None or buf.shape != obj.shape or buf.dtype != obj.dtype:
                pin = obj.is_cuda and torch.cuda.is_available()
                buf = torch.empty(obj.shape, dtype=obj.dtype, device="cpu", pin_memory=pin)
                self._buffers[path] = buf
            buf.copy_(obj.detach(), non_blocking=True)
            return buf
        if isinstance(obj, dict):
            return {k: self._copy(v, f"{path}/{k}") for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._copy(v, f"{path}/{i}") for i, v in enumerate(obj))
        return obj

    def save(self, step: int) -> None:
        began = time.time()
        waited = 0.0
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
            waited = time.time() - began
        snapshot = {
            "model": self._copy(self.model.state_dict(), "model"),
            "optimizer": self._copy(self.optimizer.state_dict(), "optimizer"),
            "step": step,
            "measured": self.measured.state(),
        }
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stall = time.time() - began
        self.measured.update("stall", stall - waited)
        self.record.write("stall_end", step=step, stall=stall, waited=waited)
        self._writing.set()
        self._thread = threading.Thread(target=self._write, args=(step, snapshot), daemon=True)
        self._thread.start()
        self._plan_next(step)

    def _write(self, step: int, snapshot: dict) -> None:
        began = time.time()
        name = f"ckpt-{step:08d}.pt"
        tmp = self.dir / (name + ".tmp")
        with open(tmp, "wb") as f:
            w = _CrcWriter(f)
            torch.save(snapshot, w)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.dir / name)
        _write_json_atomic(self.dir / MANIFEST, {"step": step, "file": name, "bytes": w.bytes, "crc": w.crc})
        self._writing.clear()
        self._after_write(step, began, {"bytes": w.bytes, "crc": w.crc})
        for old in sorted(self.dir.glob("ckpt-*.pt"))[: -self.keep]:
            old.unlink()

    def restore(self) -> int:
        """Load the newest finished checkpoint; return its step (0 if none)."""
        began = time.time()
        info = self.manifest()
        if info is None:
            self.record.write("restore", step=0, seconds=0.0, found=False)
            return 0
        path = self.dir / info["file"]
        ok = file_crc(path) == info["crc"]
        if not ok:
            raise RuntimeError(f"checksum mismatch in {path}")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.measured.load(state["measured"])
        seconds = time.time() - began
        self.measured.update("restore", seconds)
        step = state["step"]
        self.record.write("restore", step=step, seconds=seconds, found=True, crc_ok=ok)
        self._plan_next(step)
        return step

    def finish(self) -> None:
        if self._thread is not None:
            self._thread.join()


class DCPCheckpointer(_Base):
    """Baseline: PyTorch Distributed Checkpoint `async_save` at a fixed interval.

    `async_save` copies the state to CPU memory, returns, and writes in the
    background. One save is kept in flight, as in the two-phase library.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import torch.distributed as dist

        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
            dist.init_process_group("gloo", rank=0, world_size=1)
        self._future = None

    def _state(self) -> dict:
        from torch.distributed.checkpoint.state_dict import get_state_dict

        model_sd, optim_sd = get_state_dict(self.model, self.optimizer)
        return {"model": model_sd, "optimizer": optim_sd}

    def save(self, step: int) -> None:
        import torch.distributed.checkpoint as dcp

        began = time.time()
        waited = 0.0
        if self._future is not None and not self._future.done():
            self._future.result()
            waited = time.time() - began
        name = f"dcp-{step:08d}"
        write_began = time.time()
        self._writing.set()
        future = dcp.async_save(self._state(), checkpoint_id=str(self.dir / name))
        stall = time.time() - began
        self.measured.update("stall", stall - waited)
        self.record.write("stall_end", step=step, stall=stall, waited=waited)

        def done(fut, step=step, name=name, began=write_began):
            fut.result()
            _write_json_atomic(self.dir / MANIFEST, {"step": step, "file": name, "measured": self.measured.state()})
            self._writing.clear()
            self._after_write(step, began, {})

        future.add_done_callback(done)
        self._future = future
        self._plan_next(step)

    def restore(self) -> int:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_state_dict

        began = time.time()
        info = self.manifest()
        if info is None:
            self.record.write("restore", step=0, seconds=0.0, found=False)
            return 0
        state = self._state()
        dcp.load(state, checkpoint_id=str(self.dir / info["file"]))
        set_state_dict(self.model, self.optimizer,
                       model_state_dict=state["model"], optim_state_dict=state["optimizer"])
        self.measured.load(info["measured"])
        seconds = time.time() - began
        self.measured.update("restore", seconds)
        self.record.write("restore", step=info["step"], seconds=seconds, found=True)
        self._plan_next(info["step"])
        return info["step"]

    def finish(self) -> None:
        if self._future is not None:
            self._future.result()


def make_checkpointer(spec: str, directory, model, optimizer, record: Record,
                      policy_from_spec: Callable[[str], Policy]) -> _Base:
    policy = policy_from_spec(spec)
    cls = DCPCheckpointer if spec.startswith("dcp:") else TwoPhaseCheckpointer
    return cls(directory, model, optimizer, policy, record)
