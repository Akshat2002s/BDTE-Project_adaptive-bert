import numpy as np

from abert.controller import Action, Traffic, pareto_actions, sla_greedy, solve_mdp
from abert.sim import ServedPoint, mmpp_arrivals, simulate


def actions():
    mk = lambda n, acc, s: Action(n, acc, np.array([s]), np.array([1.0]))
    return [mk("fast", 0.80, 0.02), mk("mid", 0.88, 0.06), mk("slow", 0.91, 0.13),
            mk("dominated", 0.85, 0.10)]


def test_pareto_drops_dominated():
    assert [a.name for a in pareto_actions(actions())] == ["fast", "mid", "slow"]


def test_mdp_switches_to_cheaper_points_as_queue_grows():
    acts = pareto_actions(actions())
    traffic = Traffic.two_state(4.0, 15.0, 20.0, 5.0)
    pol = solve_mdp(acts, traffic, capacity=30, sla_s=0.5, beta=1.0, drop_penalty=1.0)
    service = [acts[c].mean_s for c in pol.choice[1:]]
    assert service[-1] == min(a.mean_s for a in acts)       # long queue: cheapest
    assert all(a >= b for a, b in zip(service, service[1:]))  # monotone
    # light traffic: an empty system gets the most accurate point
    light = solve_mdp(acts, Traffic.two_state(1.0, 2.0, 20.0, 5.0), 30, 0.5, 0.2, 1.0)
    assert light(1).name == "slow"
    # no waiting cost -> always the most accurate point unless chance of drop is high
    lazy = solve_mdp(acts, Traffic.two_state(1.0, 1.0, 1.0, 1.0), 30, 0.5, 0.0, 1.0)
    assert lazy(1).name == "slow"


def test_greedy_rule():
    choose = sla_greedy(pareto_actions(actions()), sla_s=0.5)
    assert choose(1).name == "slow" and choose(5).name == "mid" and choose(100).name == "fast"


def test_simulation_accounts_for_every_query():
    rng = np.random.default_rng(0)
    arr = mmpp_arrivals((4.0, 30.0), (10.0, 5.0), 200.0, rng)
    pt = ServedPoint("slow", np.full(100, 12), np.ones(100, bool), {12: 0.13}, {12: 0.0}, {12: 11.0})
    res = simulate(lambda q: pt, {"slow": pt}, arr, capacity=10, sla_s=0.5,
                   rng=np.random.default_rng(1))
    assert res["served"] + round(res["drop_rate"] / 100 * len(arr)) == len(arr)
    assert res["accuracy"] == 100.0 and res["drop_rate"] > 0
    assert abs(res["resp_ms_p95"]) <= 0.13 * 1000 * 10 + 1e-6   # never more than a full queue
