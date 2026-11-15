"""Cost model for two-phase checkpointing and the interval rules built on it.

All times are in seconds. The lost fraction of training time for a
checkpoint interval T is

    L(T) = C / T + lam * (tau + T / 2 + R)

where C is the cost of one save (stall plus the step slowdown during the
background write), tau the background write time, R the restore time and
lam = 1 / M the failure rate. The first term is paid on every save; the
second is paid on every failure: the newest durable checkpoint is on
average tau + T/2 old when a failure hits, and that work is redone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# CheckFreq keeps at most one background write in flight. The next save may
# start only after the previous write is done, with a 20% margin.
IN_FLIGHT_MARGIN = 1.2


@dataclass(frozen=True)
class SaveCosts:
    """Measured inputs of the cost model, all in seconds."""

    step: float  # time of one training step without a write running
    stall: float  # training loop blocked while the state is copied to CPU memory
    write: float  # background write time (tau)
    slowdown: float  # fractional step slowdown while a write runs (0.03 = 3%)
    restore: float  # restore time after a failure (R)

    @property
    def save_cost(self) -> float:
        """C: what one save costs the training loop."""
        return self.stall + self.slowdown * self.write

    @property
    def min_interval(self) -> float:
        """Shortest interval that keeps one write in flight."""
        return IN_FLIGHT_MARGIN * (self.stall + self.write)


def lost_fraction(interval: float, costs: SaveCosts, failure_rate: float) -> float:
    """L(T): fraction of wall-clock time lost to checkpointing and failures."""
    if interval <= 0:
        raise ValueError("interval must be positive")
    redo_age = costs.write + interval / 2 + costs.restore
    return costs.save_cost / interval + failure_rate * redo_age


def lost_seconds_per_hour(interval: float, costs: SaveCosts, failure_rate: float) -> float:
    return 3600.0 * lost_fraction(interval, costs, failure_rate)


def young_interval(costs: SaveCosts, failure_rate: float) -> float:
    """Young's continuous optimum T* = sqrt(2 C M), before any bound."""
    return math.sqrt(2.0 * costs.save_cost / failure_rate)


def failure_aware_steps(costs: SaveCosts, failure_rate: float) -> int:
    """Interval in whole steps that minimises L(T), never below the in-flight bound.

    L is convex in T, so only the two whole-step neighbours of Young's
    optimum (or the bound itself) need checking.
    """
    k_min = math.ceil(costs.min_interval / costs.step)
    k_star = young_interval(costs, failure_rate) / costs.step
    candidates = {max(k_min, math.floor(k_star)), max(k_min, math.ceil(k_star))}
    return min(candidates, key=lambda k: lost_fraction(k * costs.step, costs, failure_rate))


def checkfreq_steps(costs: SaveCosts, budget: float) -> int:
    """CheckFreq's rule: stall / budget, raised to the in-flight bound, in whole steps.

    The failure rate is not an input; that is the gap this project tests.
    """
    if not 0 < budget < 1:
        raise ValueError("budget is a fraction, for example 0.01 for 1%")
    seconds = max(costs.stall / budget, costs.min_interval)
    return math.ceil(seconds / costs.step)
