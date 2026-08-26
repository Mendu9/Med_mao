"""Contract tests for the provider-neutral ModelRegistry and model gateway.

These tests pin the behaviour that fixes P1-17 (nested-default downgrade hazard)
and P1-18 (retired model IDs reaching the provider).
"""
from __future__ import annotations

import pytest

from mao.providers.registry import (
    ModelRecord,
    ModelRegistry,
    ModelRole,
    ModelStatus,
    Modality,
)


def _record(model_id: str, *, status: ModelStatus = ModelStatus.ACTIVE, **kw) -> ModelRecord:
    return ModelRecord(
        provider=kw.pop("provider", "groq"),
        model_id=model_id,
        modality=kw.pop("modality", Modality.TEXT),
        context_limit=kw.pop("context_limit", 8192),
        status=status,
        **kw,
    )


class TestModelRoleResolution:
    def test_resolves_every_declared_role_to_an_active_record(self) -> None:
        registry = ModelRegistry.default()
        for role in ModelRole:
            record = registry.resolve(role)
            assert record.status is ModelStatus.ACTIVE, (
                f"role {role.value} resolves to non-active model {record.model_id}"
            )
            assert record.model_id, f"role {role.value} resolved to an empty model id"

    def test_model_id_for_returns_the_resolved_id(self) -> None:
        registry = ModelRegistry.default()
        assert registry.model_id_for(ModelRole.ROUTER_FAST) == registry.resolve(
            ModelRole.ROUTER_FAST
        ).model_id


class TestRetiredModelRefusal:
    """P1-18 — a retired model must never be handed to a provider."""

    def test_resolve_falls_back_when_the_primary_record_is_retired(self) -> None:
        registry = ModelRegistry(
            records={
                "retired-a": _record("retired-a", status=ModelStatus.RETIRED, fallbacks=("live-b",)),
                "live-b": _record("live-b"),
            },
            role_bindings={ModelRole.VISION: "retired-a"},
        )
        assert registry.resolve(ModelRole.VISION).model_id == "live-b"

    def test_resolve_raises_when_no_active_fallback_exists(self) -> None:
        registry = ModelRegistry(
            records={"retired-only": _record("retired-only", status=ModelStatus.RETIRED)},
            role_bindings={ModelRole.VISION: "retired-only"},
        )
        with pytest.raises(LookupError, match="no active model"):
            registry.resolve(ModelRole.VISION)

    def test_known_retired_groq_ids_are_recorded_as_retired(self) -> None:
        registry = ModelRegistry.default()
        for retired in (
            "llama-3.2-11b-vision-preview",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
        ):
            record = registry.get(retired)
            assert record is not None, f"{retired} missing from registry"
            assert record.status is ModelStatus.RETIRED

    def test_no_role_binds_directly_to_a_retired_id(self) -> None:
        registry = ModelRegistry.default()
        for role in ModelRole:
            bound_id = registry.binding_for(role)
            assert registry.records[bound_id].status is ModelStatus.ACTIVE


class TestSafetyRoleIsolation:
    """P1-17 — a generic GROQ_MODEL override must not downgrade safety roles."""

    def test_generic_model_env_does_not_alter_safety_roles(self, monkeypatch) -> None:
        monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
        monkeypatch.delenv("CLINICAL_MODEL", raising=False)
        monkeypatch.delenv("MAO_MODEL_CLINICAL_SYNTHESIS", raising=False)
        monkeypatch.delenv("MAO_MODEL_SAFETY_JUDGE", raising=False)

        registry = ModelRegistry.default()

        assert registry.model_id_for(ModelRole.CLINICAL_SYNTHESIS) != "llama-3.1-8b-instant"
        assert registry.model_id_for(ModelRole.SAFETY_JUDGE) != "llama-3.1-8b-instant"

    def test_each_role_reads_its_own_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("MAO_MODEL_SAFETY_JUDGE", "llama-3.3-70b-versatile")
        registry = ModelRegistry.default()
        assert registry.model_id_for(ModelRole.SAFETY_JUDGE) == "llama-3.3-70b-versatile"

    def test_env_override_naming_all_roles(self) -> None:
        for role in ModelRole:
            assert role.env_var == f"MAO_MODEL_{role.name}"


class TestRecordMetadata:
    def test_records_carry_the_architecture_mandated_fields(self) -> None:
        record = ModelRegistry.default().resolve(ModelRole.GENERAL_SYNTHESIS)
        for field_name in (
            "provider",
            "model_id",
            "modality",
            "context_limit",
            "supports_structured_output",
            "supports_tools",
            "status",
            "fallbacks",
            "cost_per_1m_input_usd",
            "cost_per_1m_output_usd",
            "last_verified_at",
        ):
            assert hasattr(record, field_name), f"ModelRecord missing {field_name}"

    def test_vision_role_resolves_to_a_vision_capable_record(self) -> None:
        record = ModelRegistry.default().resolve(ModelRole.VISION)
        assert record.modality is Modality.VISION
