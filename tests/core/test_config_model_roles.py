"""config.py must delegate model selection to the registry.

Pins the fix for P1-17. Note that `.env` sets CLINICAL_MODEL, and `load_dotenv`
re-applies it on every module reload — so a test that merely deletes the env var
proves nothing. These tests assert the *delegation property* instead: the legacy
constants must track the role bindings, including per-role overrides the old
`getenv` chain knows nothing about.
"""
from __future__ import annotations

import importlib

from mao.providers.gateway import model_id_for, reset_registry
from mao.providers.registry import ModelRole


def _reload_config():
    import mao.core.config as config

    reset_registry()
    return importlib.reload(config)


class TestLegacyAliasesTrackRoleBindings:
    def test_fast_model_tracks_general_synthesis_role(self) -> None:
        config = _reload_config()
        assert config.FAST_MODEL == model_id_for(ModelRole.GENERAL_SYNTHESIS)

    def test_clinical_model_tracks_clinical_synthesis_role(self) -> None:
        config = _reload_config()
        assert config.CLINICAL_MODEL == model_id_for(ModelRole.CLINICAL_SYNTHESIS)

    def test_per_role_override_reaches_the_legacy_clinical_alias(self, monkeypatch) -> None:
        """The registry's per-role env var must win, not the legacy chain."""
        monkeypatch.setenv("MAO_MODEL_CLINICAL_SYNTHESIS", "llama-3.1-8b-instant")
        config = _reload_config()
        assert config.CLINICAL_MODEL == "llama-3.1-8b-instant"

    def test_per_role_override_reaches_the_legacy_fast_alias(self, monkeypatch) -> None:
        monkeypatch.setenv("MAO_MODEL_GENERAL_SYNTHESIS", "llama-3.3-70b-versatile")
        config = _reload_config()
        assert config.FAST_MODEL == "llama-3.3-70b-versatile"

    def test_judge_model_tracks_safety_judge_role(self) -> None:
        config = _reload_config()
        assert config.cfg.groq_judge_model == model_id_for(ModelRole.SAFETY_JUDGE)


class TestSafetyConstantsComeFromPolicy:
    def test_mri_gate_matches_policy(self) -> None:
        config = _reload_config()
        from mao.safety.policy import get_policy

        assert config.MRI_CONFIDENCE_GATE is get_policy().mri_confidence_gate


class TestDeadConfigRemoved:
    def test_code_exec_timeout_is_gone(self) -> None:
        """P2-5: the code agent was removed; its timeout config is dead."""
        assert not hasattr(_reload_config().cfg, "code_exec_timeout")

    def test_ollama_base_url_is_gone(self) -> None:
        """P2-5: no executable code uses Ollama."""
        assert not hasattr(_reload_config().cfg, "ollama_base_url")


def teardown_module() -> None:
    """Leave the registry and config module in their default state."""
    reset_registry()
    import mao.core.config as config

    importlib.reload(config)
