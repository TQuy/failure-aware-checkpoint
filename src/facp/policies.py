"""Interval policies: when to take the next checkpoint.

Every policy answers one question after each finished write: how many steps
until the next save. The adaptive policies read the measured costs from
`Measured`, which the checkpointer updates from its own timings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from facp.cost_model import SaveCosts, checkfreq_steps, failure_aware_steps


@dataclass
class Measured:
    """Running averages of the cost-model inputs, in seconds."""

    alpha: float = 0.3  # weight of the newest sample
    step_clean: float | None = None
    step_writing: float | None = None
    stall: float | None = None
    write: float | None = None
    restore: float = 0.0
    # The first save of a process stalls about 4.9 s against 0.75 s later on
    # GPT-2 large. Kept in the running average, it held CheckFreq 1% at
    # 80-110 steps for the first 25 minutes of each run, so it is dropped.
    skip_first: tuple = ("stall",)
    _seen: set = field(default_factory=set)
    _skipped: set = field(default_factory=set)

    def update(self, name: str, value: float) -> None:
        if name in self.skip_first and name not in self._skipped:
            self._skipped.add(name)
            return
        old = getattr(self, name)
        new = value if old is None or name not in self._seen else (1 - self.alpha) * old + self.alpha * value
        setattr(self, name, new)
        self._seen.add(name)

    def ready(self) -> bool:
        return None not in (self.step_clean, self.stall, self.write)

    def costs(self) -> SaveCosts:
        writing = self.step_writing if self.step_writing is not None else self.step_clean
        slowdown = max(0.0, writing / self.step_clean - 1.0)
        return SaveCosts(
            step=self.step_clean,
            stall=self.stall,
            write=self.write,
            slowdown=slowdown,
            restore=self.restore,
        )

    def state(self) -> dict:
        return {k: getattr(self, k) for k in ("step_clean", "step_writing", "stall", "write", "restore")}

    def load(self, state: dict) -> None:
        for k, v in state.items():
            setattr(self, k, v)
            if v is not None:
                self._seen.add(k)


class Policy:
    name = "policy"

    def __init__(self, first_steps: int):
        self.first_steps = first_steps

    def interval_steps(self, measured: Measured) -> int:
        raise NotImplementedError


class Fixed(Policy):
    def __init__(self, steps: int):
        super().__init__(steps)
        self.name = f"fixed-{steps}"

    def interval_steps(self, measured: Measured) -> int:
        return self.first_steps


class CheckFreq(Policy):
    """Interval from an overhead budget (Mohan et al., FAST 2021)."""

    def __init__(self, budget: float, first_steps: int = 10):
        super().__init__(first_steps)
        self.budget = budget
        self.name = f"checkfreq-{budget:g}"

    def interval_steps(self, measured: Measured) -> int:
        if not measured.ready():
            return self.first_steps
        return checkfreq_steps(measured.costs(), self.budget)


class FailureAware(Policy):
    """Interval that minimises L(T) at the given failure rate."""

    def __init__(self, mtbf_seconds: float, first_steps: int = 10):
        super().__init__(first_steps)
        self.failure_rate = 1.0 / mtbf_seconds
        self.name = f"failure-aware-{mtbf_seconds:g}"

    def interval_steps(self, measured: Measured) -> int:
        if not measured.ready():
            return self.first_steps
        return failure_aware_steps(measured.costs(), self.failure_rate)


def from_spec(spec: str) -> Policy:
    """Parse 'fixed:40', 'checkfreq:0.01', 'failure-aware:3600' or 'dcp:40'."""
    kind, _, arg = spec.partition(":")
    if kind in ("fixed", "dcp"):
        return Fixed(int(arg))
    if kind == "checkfreq":
        return CheckFreq(float(arg))
    if kind == "failure-aware":
        return FailureAware(float(arg))
    raise ValueError(f"unknown policy {spec!r}")
