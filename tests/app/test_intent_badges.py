"""The UI must not keep its own copy of the backend's intent list.

Architecture: "UI must not manually duplicate backend topology, intent lists,
model lists, or health schemas when a canonical registry can provide them."
The badge map had drifted to include `code` and `sql`, both of which were
removed from the backend (P2-3, P0-3).
"""
from __future__ import annotations

from mao.core.state import ALL_INTENTS


class TestIntentBadgeColours:
    def test_every_backend_intent_has_a_badge_colour(self) -> None:
        from app.intent_badges import colour_for

        for intent in ALL_INTENTS:
            assert colour_for(intent)

    def test_no_badge_exists_for_an_intent_the_backend_does_not_have(self) -> None:
        from app.intent_badges import INTENT_COLOURS

        assert set(INTENT_COLOURS) <= set(ALL_INTENTS), (
            f"UI advertises dead intents: {sorted(set(INTENT_COLOURS) - set(ALL_INTENTS))}"
        )

    def test_removed_intents_are_absent(self) -> None:
        from app.intent_badges import INTENT_COLOURS

        assert "code" not in INTENT_COLOURS
        assert "sql" not in INTENT_COLOURS

    def test_unknown_intent_falls_back_to_a_neutral_colour(self) -> None:
        from app.intent_badges import colour_for

        assert colour_for("something-new") == "gray"

    def test_empty_intent_falls_back(self) -> None:
        from app.intent_badges import colour_for

        assert colour_for("") == "gray"
