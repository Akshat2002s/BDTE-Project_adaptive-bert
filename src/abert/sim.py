"""Discrete-event simulation of the CPU gateway: one server, batch size 1, FIFO.

Arrivals follow a two-state Markov-modulated Poisson process (calm / burst).
Each served query is a random test-set query: the chosen operating point's real
exit layer and correctness for that query (from measure_elastic.py) decide its
service time and whether it was answered correctly, so accuracy and latency stay
correlated the way they are in the real model. Service times get log-normal
jitter fitted to the measured p50 / p95.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ServedPoint:
    """What the simulator needs to serve one query at an operating point."""
    name: str
    exit_layer: np.ndarray        # (n_test,) exit layer per test query
    correct: np.ndarray           # (n_test,) bool
    lat_p50_s: dict[int, float]   # exit layer -> median service time (s)
    lat_sigma: dict[int, float]   # exit layer -> log-normal sigma
    gflops: dict[int, float]      # exit layer -> GFLOPs actually spent


def mmpp_arrivals(rates: tuple[float, float], mean_durations: tuple[float, float],
                  duration_s: float, rng: np.random.Generator) -> np.ndarray:
    times, t, regime = [], 0.0, 0
    while t < duration_s:
        end = min(t + rng.exponential(mean_durations[regime]), duration_s)
        n = rng.poisson(rates[regime] * (end - t))
        times.append(rng.uniform(t, end, n))
        t, regime = end, 1 - regime
    return np.sort(np.concatenate(times))


def simulate(policy, points: dict[str, ServedPoint], arrivals: np.ndarray, capacity: int,
             sla_s: float, rng: np.random.Generator) -> dict[str, float]:
    """`policy(q)` returns an object with a `.name` that is a key of `points`."""
    queue: deque[float] = deque()
    free_at, i, n = 0.0, 0, len(arrivals)
    resp, correct, flops, dropped, choices = [], [], [], 0, {}
    n_test = len(next(iter(points.values())).correct)
    while i < n or queue:
        start = max(free_at, queue[0]) if queue else None
        if i < n and (start is None or arrivals[i] <= start):
            ta = arrivals[i]
            i += 1
            if len(queue) + (free_at > ta) >= capacity:
                dropped += 1
            else:
                queue.append(ta)
            continue
        pt = points[policy(len(queue)).name]
        choices[pt.name] = choices.get(pt.name, 0) + 1
        j = rng.integers(n_test)
        e = int(pt.exit_layer[j])
        service = pt.lat_p50_s[e] * float(np.exp(pt.lat_sigma[e] * rng.standard_normal()))
        free_at = start + service
        resp.append(free_at - queue.popleft())
        correct.append(bool(pt.correct[j]))
        flops.append(pt.gflops[e])
    resp = np.array(resp)
    total = len(resp) + dropped
    return {
        "accuracy": 100.0 * float(np.mean(correct)) if correct else 0.0,
        "effective_accuracy": 100.0 * float(np.sum(correct)) / total,   # drops count as wrong
        "drop_rate": 100.0 * dropped / total,
        "sla_violation": 100.0 * (float(np.sum(resp > sla_s)) + dropped) / total,
        "resp_ms_mean": 1000.0 * float(resp.mean()),
        "resp_ms_p95": 1000.0 * float(np.percentile(resp, 95)),
        "gflops_mean": float(np.mean(flops)),
        "served": len(resp),
        "choices": choices,
    }