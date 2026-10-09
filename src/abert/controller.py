"""Queue-aware MDP controller that picks the operating point for each query.

Model (a semi-Markov decision process, solved by value iteration):
  * decision epoch: the server becomes free and at least one query is waiting
  * state q:        queries in the system at that moment (1 .. capacity)
  * action a:       an operating point (width + exit rule) with a known val
                    accuracy and a service-time distribution (one latency per
                    exit, weighted by how often queries leave at each exit)
  * arrivals:       during a service of length S, A ~ Poisson(lambda_r * S),
                    mixed over the traffic regimes r (calm / burst) with their
                    stationary probabilities. That keeps the state 1-D but makes
                    the arrival count as bursty as the real traffic.
  * next state:     min(q - 1 + A, capacity); arrivals beyond capacity are dropped
  * reward:         accuracy(a)
                    - beta * expected waiting accumulated during this service / SLA
                    - drop_penalty * expected drops
Accuracies come from the val split; the simulation scores on test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


@dataclass
class Action:
    name: str
    accuracy: float               # val accuracy as a fraction (0..1)
    service_s: np.ndarray         # possible service times in seconds (one per exit)
    service_p: np.ndarray         # probability of each service time
    gflops: float = 0.0

    @property
    def mean_s(self) -> float:
        return float(self.service_p @ self.service_s)

    @property
    def second_moment_s(self) -> float:
        return float(self.service_p @ self.service_s ** 2)


def pareto_actions(actions: list[Action]) -> list[Action]:
    """Keep actions not dominated in (lower mean service time, higher accuracy),
    sorted from cheapest to most accurate."""
    keep, best = [], -1.0
    for a in sorted(actions, key=lambda a: (a.mean_s, -a.accuracy)):
        if a.accuracy > best:
            keep.append(a)
            best = a.accuracy
    return keep


@dataclass
class Traffic:
    rates: np.ndarray             # arrival rate per regime (queries / s)
    probs: np.ndarray             # stationary probability of each regime

    @property
    def mean_rate(self) -> float:
        return float(self.rates @ self.probs)

    @classmethod
    def two_state(cls, calm_rate: float, burst_rate: float, calm_mean_s: float,
                  burst_mean_s: float) -> "Traffic":
        p_burst = burst_mean_s / (calm_mean_s + burst_mean_s)
        return cls(np.array([calm_rate, burst_rate]), np.array([1 - p_burst, p_burst]))


def arrival_pmf(action: Action, traffic: Traffic, kmax: int) -> np.ndarray:
    """P(k arrivals during one service), k = 0..kmax (last entry = kmax or more)."""
    k = np.arange(kmax + 1)
    pmf = np.zeros(kmax + 1)
    for s, ps in zip(action.service_s, action.service_p):
        for lam, pr in zip(traffic.rates, traffic.probs):
            pmf += ps * pr * poisson.pmf(k, lam * s)
    pmf[-1] += max(0.0, 1.0 - pmf.sum())
    return pmf


@dataclass
class Policy:
    actions: list[Action]
    choice: np.ndarray            # choice[q] = action index for q queries in system (q >= 1)
    values: np.ndarray

    def __call__(self, q: int) -> Action:
        return self.actions[self.choice[min(max(q, 1), len(self.choice) - 1)]]


def solve_mdp(actions: list[Action], traffic: Traffic, capacity: int, sla_s: float,
              beta: float, drop_penalty: float, gamma: float = 0.98,
              tol: float = 1e-9, max_iter: int = 100_000) -> Policy:
    """Value iteration. Returns the optimal stationary policy q -> action."""
    nq, na = capacity + 1, len(actions)
    kmax = capacity + 1
    pmfs = [arrival_pmf(a, traffic, kmax) for a in actions]
    lam = traffic.mean_rate

    reward = np.zeros((nq, na))
    trans = np.zeros((na, nq, nq))            # trans[a, q, q']
    for j, (a, pmf) in enumerate(zip(actions, pmfs)):
        for q in range(1, nq):
            wait = q * a.mean_s + lam * a.second_moment_s / 2   # query-seconds of waiting
            nxt = q - 1 + np.arange(kmax + 1)
            drops = np.maximum(0, nxt - capacity) @ pmf
            reward[q, j] = a.accuracy - beta * wait / sla_s - drop_penalty * drops
            np.add.at(trans[j, q], np.minimum(nxt, capacity), pmf)
    # q = 0: server idles until the next arrival, then q = 1 (no action, no reward)
    v = np.zeros(nq)
    for _ in range(max_iter):
        qv = reward + gamma * np.einsum("aqr,r->qa", trans, v)
        new = qv.max(1)
        new[0] = gamma * v[1]
        if np.abs(new - v).max() < tol:
            v = new
            break
        v = new
    choice = qv.argmax(1)
    choice[0] = choice[1]
    return Policy(actions, choice, v)


def sla_greedy(actions: list[Action], sla_s: float):
    """Baseline heuristic: the most accurate action whose mean service time, times
    the queue in front of it, still fits the SLA; otherwise the cheapest one."""
    ordered = sorted(actions, key=lambda a: -a.accuracy)
    cheapest = min(actions, key=lambda a: a.mean_s)

    def choose(q: int) -> Action:
        for a in ordered:
            if q * a.mean_s <= sla_s:
                return a
        return cheapest
    return choose