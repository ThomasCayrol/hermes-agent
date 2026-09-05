"""Regression test for TUI v2 blitz bug: explicit /model --provider switch
silently fell back to the old primary provider on the next turn because the
fallback chain — seeded from config at agent __init__ — kept entries for the
provider the user just moved away from.

Reported: "switched from openrouter provider to anthropic api key via hermes
model and the tui keeps trying openrouter".

Extended with regression coverage for the inverse failure (provider-fallback
preserve on primary return): a session re-pin back to the primary runtime
through ``switch_model`` must NOT prune the configured fallback entry that
matches the provider being switched away from — otherwise the next HTTP 429
on the restored primary reaches the surface with no fallback attempted.
"""

from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent


def _make_agent(chain):
    agent = AIAgent.__new__(AIAgent)

    agent.provider = "openrouter"
    agent.model = "x-ai/grok-4"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "or-key"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent._client_kwargs = {"api_key": "or-key", "base_url": "https://openrouter.ai/api/v1"}
    agent.context_compressor = None
    agent._anthropic_api_key = ""
    agent._anthropic_base_url = None
    agent._anthropic_client = None
    agent._is_anthropic_oauth = False
    agent._cached_system_prompt = "cached"
    agent._primary_runtime = {}
    agent._fallback_activated = False
    agent._fallback_index = 0
    agent._fallback_chain = list(chain)
    agent._fallback_model = chain[0] if chain else None

    return agent


def _switch_to_anthropic(agent):
    with (
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
        patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-xyz"),
        patch("agent.anthropic_adapter._is_oauth_token", return_value=False),
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
    ):
        agent.switch_model(
            new_model="claude-sonnet-4-5",
            new_provider="anthropic",
            api_key="sk-ant-xyz",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
        )


def test_switch_drops_old_primary_from_fallback_chain():
    agent = _make_agent([
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ])

    _switch_to_anthropic(agent)

    providers = [entry["provider"] for entry in agent._fallback_chain]

    assert "openrouter" not in providers, "old primary must be pruned"
    assert "anthropic" not in providers, "new primary is redundant in the chain"
    assert providers == ["nous"]
    assert agent._fallback_model == {"provider": "nous", "model": "hermes-4"}


def test_switch_with_empty_chain_stays_empty():
    agent = _make_agent([])

    _switch_to_anthropic(agent)

    assert agent._fallback_chain == []
    assert agent._fallback_model is None


def test_manual_switch_clears_provider_fallback_provenance():
    agent = _make_agent([
        {"provider": "openrouter", "model": "x-ai/grok-4"},
        {"provider": "nous", "model": "hermes-4"},
    ])
    agent._provider_fallback_active = True
    agent._provider_fallback_route = ("fallback-model", "fallback-provider")

    _switch_to_anthropic(agent)

    assert agent._provider_fallback_active is False
    assert agent._provider_fallback_route is None




def test_switch_within_same_provider_preserves_chain():
    chain = [{"provider": "openrouter", "model": "x-ai/grok-4"}]
    agent = _make_agent(chain)

    with patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None):
        agent.switch_model(
            new_model="openai/gpt-5",
            new_provider="openrouter",
            api_key="or-key",
            base_url="https://openrouter.ai/api/v1",
        )

    assert agent._fallback_chain == chain


# ── Return-to-primary preserves the fallback chain ─────────────────────────
#
# Observed incident: an agent on DeepSeek (automatic fallback from a
# gpt-5.6-luna / openai-codex primary) was re-pinned back to the primary
# runtime through switch_model.  The provider-change pruning dropped the
# DeepSeek entry because it matched the provider being switched away from,
# leaving no fallback for the next HTTP 429.  These tests lock in the
# carve-out: returning to the provider recorded in ``_primary_runtime`` is
# not a rejection of the fallback provider.

def _make_full_agent(chain):
    """Full AIAgent (real init) with OpenAI client construction stubbed."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="luna-key",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=chain,
        )
    agent.client = MagicMock()
    return agent


def _mock_client(base_url="https://api.deepseek.com/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def test_return_to_primary_preserves_fallback_chain():
    """Case A: Luna → DeepSeek (fallback) → back to Luna keeps DeepSeek.

    The agent sits on the DeepSeek fallback (as after an automatic
    failover); its primary snapshot still points at gpt-5.6-luna via
    openai-codex.  Re-pinning through switch_model must NOT prune the
    DeepSeek fallback entry.
    """
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = _make_full_agent(chain)

    # Post-fallback state: runtime is DeepSeek, primary snapshot is Luna.
    agent.model = "deepseek-v4-flash"
    agent.provider = "deepseek"
    agent.api_mode = "chat_completions"
    agent._primary_runtime = {
        "model": "gpt-5.6-luna",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "api_mode": "codex_responses",
    }
    agent._fallback_activated = True

    with (
        patch("hermes_cli.timeouts.get_provider_request_timeout", return_value=None),
        patch("run_agent.OpenAI"),
    ):
        agent.switch_model(
            new_model="gpt-5.6-luna",
            new_provider="openai-codex",
            api_key="luna-key",
            base_url="https://chatgpt.com/backend-api/codex",
            api_mode="codex_responses",
        )

    assert agent.provider == "openai-codex"
    assert agent.model == "gpt-5.6-luna"
    providers = [entry["provider"] for entry in agent._fallback_chain]
    assert providers == ["deepseek"], (
        "returning to the primary runtime must not prune the fallback entry"
    )
    assert agent._fallback_model == chain[0]
    assert agent._fallback_index == 0


def test_return_to_primary_then_429_fails_over_to_fallback():
    """Case B: after the return to the primary, an HTTP 429 still fails over.

    Chain continuation of the re-pin regression: the preserved DeepSeek
    entry must actually be attempted when the restored primary hits a
    rate limit.
    """
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = _make_full_agent(chain)

    agent.model = "gpt-5.6-luna"
    agent.provider = "openai-codex"
    agent._fallback_index = 0
    agent._fallback_activated = False

    with patch(
        "agent.auxiliary_client.resolve_provider_client",
        return_value=(_mock_client(), "deepseek-v4-flash"),
    ):
        assert agent._try_activate_fallback(FailoverReason.rate_limit) is True

    assert agent.provider == "deepseek"
    assert agent.model == "deepseek-v4-flash"
    assert agent._fallback_activated is True
    assert agent._fallback_index == 1


def test_switch_away_from_fallback_to_other_provider_still_prunes():
    """Case C: pruning still applies when the destination is NOT the primary.

    Leaving the fallback provider for a brand-new primary is a real manual
    switch — the old fallback entry must still be pruned.  Only the return
    to ``_primary_runtime`` gets the carve-out.
    """
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = _make_agent(chain)

    agent.model = "deepseek-v4-flash"
    agent.provider = "deepseek"
    agent._primary_runtime = {
        "model": "gpt-5.6-luna",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }

    _switch_to_anthropic(agent)

    assert agent.provider == "anthropic"
    assert agent._fallback_chain == [], (
        "a manual switch to a provider that is not the primary still prunes"
    )
    assert agent._fallback_model is None
