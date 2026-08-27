"""Intent badge colours for the Streamlit UI.

Colour is a presentation concern and lives here; the *set* of intents is the
backend's, and is never re-listed in the UI. The previous inline map in
`streamlit_app.py` had drifted to advertise `code` and `sql` after both were
removed from the graph.
"""
from __future__ import annotations

from mao.core.state import ALL_INTENTS

_FALLBACK_COLOUR = "gray"

# Presentation preferences, keyed by intent. Anything the backend adds that is
# not listed here simply renders neutral rather than breaking the UI.
_PREFERRED: dict[str, str] = {
    "graphrag": "blue",
    "clinical": "red",
    "tool": "orange",
    "summarize": "gray",
    "multimodal": "blue",
    "critic": "orange",
    "chitchat": "green",
    "fallback": "gray",
}

# Only intents the backend actually declares get a badge.
INTENT_COLOURS: dict[str, str] = {
    intent: _PREFERRED.get(intent, _FALLBACK_COLOUR) for intent in ALL_INTENTS
}


def colour_for(intent: str) -> str:
    """Badge colour for an intent, neutral for anything unrecognised."""
    return INTENT_COLOURS.get(intent, _FALLBACK_COLOUR)
