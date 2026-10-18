# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit ChatGPT authentication stays private and independent of API keys."""

import json
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evolve_codex as ec
from torchtitan.experiments.rl.examples.tmax import layout


def test_explicit_chatgpt_auth_is_private_and_cleaned(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "WEB_SEARCH", "")
    root = layout.Root(tmp_path / "root")
    monkeypatch.setenv("TRL_BASE", str(root.path))
    rw = root.evolution.task("task-a").rewrite("harder")
    source = tmp_path / "login.json"
    auth = {"auth_mode": "chatgpt", "tokens": {"access_token": "test-token"}}
    source.write_text(json.dumps(auth))
    monkeypatch.setenv("EVOLVE_CODEX_AUTH_FILE", str(source))
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-key")
    with ec.session(rw, "agent", timeout=10) as run:
        env = ec._codex_env(run.dir)
        target = run.dir.codex_home / "auth.json"
        assert "OPENAI_API_KEY" not in env
        assert json.loads(target.read_text()) == auth
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert ec._provider_overrides() == ["model_provider=openai"]
        assert run.meta["authentication"] == "chatgpt"
    assert not target.exists()
    assert json.loads(source.read_text()) == auth


def test_api_key_provider_remains_default(monkeypatch):
    monkeypatch.delenv("EVOLVE_CODEX_AUTH_FILE", raising=False)
    assert "model_provider=oai" in ec._provider_overrides()
    assert "model_providers.oai.env_key=OPENAI_API_KEY" in ec._provider_overrides()


def test_extra_config_reaches_both_providers(monkeypatch):
    """EVOLVE_CODEX_EXTRA_CONFIG adds `-c` overrides whichever way the session
    authenticates, split on whitespace so a systemd EnvironmentFile line
    carries it."""
    monkeypatch.setattr(ec, "WEB_SEARCH", "")
    monkeypatch.setenv(
        "EVOLVE_CODEX_EXTRA_CONFIG",
        "model_providers.oai.stream_max_retries=20  model_providers.oai.stream_idle_timeout_ms=900000",
    )
    monkeypatch.delenv("EVOLVE_CODEX_AUTH_FILE", raising=False)
    monkeypatch.setattr(ec, "ACCOUNT_HOME", None)
    overrides = ec._provider_overrides()
    assert overrides[-2:] == [
        "model_providers.oai.stream_max_retries=20",
        "model_providers.oai.stream_idle_timeout_ms=900000",
    ]
    assert "model_provider=oai" in overrides
    # Pooled-account rate limits outlast the CLI's default retries.
    assert "model_providers.oai.request_max_retries=10" in overrides
    monkeypatch.setenv("EVOLVE_CODEX_AUTH_FILE", "/nonexistent")
    assert ec._provider_overrides() == [
        "model_provider=openai",
        "model_providers.oai.stream_max_retries=20",
        "model_providers.oai.stream_idle_timeout_ms=900000",
    ]
    monkeypatch.delenv("EVOLVE_CODEX_EXTRA_CONFIG")
    assert ec._extra_config() == []


def test_web_search_names_the_provider_openai_and_turns_the_tool_on(monkeypatch):
    """codex 0.149 offers its web tool to a custom provider only when it is
    named exactly OpenAI; the default turns the tool on that way."""
    monkeypatch.delenv("EVOLVE_CODEX_ACCOUNT_HOME", raising=False)
    monkeypatch.delenv("EVOLVE_CODEX_AUTH_FILE", raising=False)
    monkeypatch.setattr(ec, "ACCOUNT_HOME", None)
    monkeypatch.setattr(ec, "WEB_SEARCH", "live")
    overrides = ec._provider_overrides()
    assert "model_providers.oai.name=OpenAI" in overrides
    assert "web_search=live" in overrides
    monkeypatch.setattr(ec, "WEB_SEARCH", "")
    overrides = ec._provider_overrides()
    assert "model_providers.oai.name=openai" in overrides
    assert not any(o.startswith("web_search=") for o in overrides)
