"""_EngineStatsLogger: vLLM engine stats reported through our logger."""

import logging
from types import SimpleNamespace

from torchtitan.experiments.rl.actors import generator


def _sched(running, waiting, kv):
    return SimpleNamespace(
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=0,
        kv_cache_usage=kv,
    )


def test_one_line_per_interval_with_peak_kv_and_preemptions(monkeypatch, caplog):
    monkeypatch.setenv("RL_VLLM_STATS_INTERVAL_SEC", "60")
    clock = [0.0]
    monkeypatch.setattr(generator.time, "monotonic", lambda: clock[0])
    stats = generator._EngineStatsLogger(vllm_config=None, engine_index=3)
    stats.record(_sched(10, 2, 0.5), SimpleNamespace(num_preempted_reqs=1))
    stats.record(_sched(12, 5, 0.9), SimpleNamespace(num_preempted_reqs=2))
    stats.record(_sched(11, 4, 0.7), None)
    with caplog.at_level(logging.INFO, logger=generator.logger.name):
        clock[0] = 30.0
        stats.log()  # inside the interval: nothing
        clock[0] = 61.0
        stats.log()
        clock[0] = 125.0
        stats.log()
    lines = [r.getMessage() for r in caplog.records if "[vllm_stats]" in r.getMessage()]
    assert lines[0] == (
        "[vllm_stats] engine=3 running=11 waiting=4 kv_peak=90.0% "
        "preempted=3 preempted_total=3 window_s=61"
    )
    assert "kv_peak=0.0% preempted=0 preempted_total=3" in lines[1]
    assert len(lines) == 2


def test_engine_stats_ride_on_request_metrics(monkeypatch):
    monkeypatch.setattr(generator, "_ENGINE_STATS", {})
    stats = generator._EngineStatsLogger(vllm_config=None)
    stats.record(_sched(5, 3, 0.25), SimpleNamespace(num_preempted_reqs=4))
    stats.record(_sched(5, 1, 0.5), SimpleNamespace(num_preempted_reqs=1))
    assert generator._ENGINE_STATS == {
        "kv_cache_usage": 0.5,
        "waiting_reqs": 1,
        "preempted_total": 5,
    }
