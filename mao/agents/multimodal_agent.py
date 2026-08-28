"""Multimodal agent: image understanding via the vision role, audio transcription via Whisper.

P1-18: the vision model is resolved from the provider registry at call time
(``ModelRole.VISION``). This module previously hard-coded a preview vision model
id in three places; the provider has since retired that id, so every image
request would have failed with a 400. No literal model id may appear here — the
registry records retired ids precisely so they can never be resolved.
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path

import requests

from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.providers.registry import ModelRole
from mao.safety.policy import ATTACHMENT_KEYS

logger = logging.getLogger(__name__)

_AUDIO_SYSTEM = """\
You are reviewing a transcript of an audio recording.
Summarize the key points and answer any questions the user has about the content.
"""


def multimodal_node(state: MAOState) -> MAOState:
    """
    LangGraph node: route to image or audio handler based on metadata.
    """
    user_query: str = state["user_query"]
    memory_context: str = state.get("memory_context", "")
    metadata: dict  = state.get("metadata", {})


    modality = metadata.get("modality", "")

    # --- Auto-detect modality if not explicitly set ---
    if not modality:
        if metadata.get("image_b64") or metadata.get("image_url"):
            modality = "image"
        elif metadata.get("audio_path") or metadata.get("audio_b64"):
            # `audio_b64` was omitted here, so a base64 voice sample with no
            # path fell through to the "no media supplied" branch — even though
            # `handle_audio` decodes exactly that key.
            modality = "audio"
        else:
            modality = "text_about_media"

    # --- Dispatch ---
    if modality == "image":
        response, result_meta = _handle_image(user_query, metadata, memory_context)
    elif modality == "audio":
        response, result_meta = handle_audio(user_query, metadata, memory_context)
    else:
        response = (
            "To use the multimodal agent, please provide an image (base64 or URL) "
            "or an audio file path along with your question."
        )
        result_meta = {"modality": "none"}


    state["response"]   = response
    state["agent_used"] = "multimodal"
    # Echo the caller's metadata back MINUS the raw attachment payloads. Graph
    # state is persisted, traced and logged; carrying a base64 scan or voice
    # sample through all of that spreads patient data far beyond the one call
    # that needed it.
    state["metadata"]   = {
        **{k: v for k, v in metadata.items() if k not in ATTACHMENT_KEYS},
        **result_meta,
        "modality": modality,
    }
    return state


# ---------------------------------------------------------------------------
# Image handler
# ---------------------------------------------------------------------------

def _handle_image(
    user_query: str,
    metadata: dict,
    memory_context: str,
) -> tuple[str, dict]:
    """
    Call Groq vision to answer a question about an image.

    Accepts:
      - metadata["image_b64"]: base64-encoded image string
      - metadata["image_url"]: HTTP URL to download and encode
    """
    image_b64: str | None = metadata.get("image_b64")

    # Download if URL provided
    if not image_b64 and metadata.get("image_url"):
        try:
            resp = requests.get(metadata["image_url"], timeout=15)
            resp.raise_for_status()
            image_b64 = base64.b64encode(resp.content).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.error("Image download failed: %s", exc)
            return f"Failed to download image: {exc}", {"error": str(exc)}

    if not image_b64:
        return "No image data provided.", {"error": "no_image"}

    system_prompt = build_system_prompt(
        get_prompt("clinical.vision").template, memory_context
    )

    try:
        completion = gateway.complete(
            role=ModelRole.VISION,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ]},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return completion.text.strip(), {"vision_model": completion.model_id}
    except Exception as exc:
        logger.error("Vision call failed: %s", exc)
        return f"Image analysis failed: {exc}", {"error": str(exc)}


# ---------------------------------------------------------------------------
# Audio handler
# ---------------------------------------------------------------------------

def handle_audio(  # public: `clinical_agent` shares this one implementation
    user_query: str,
    metadata: dict,
    memory_context: str,
) -> tuple[str, dict]:
    """
    Transcribe audio with Whisper, then answer user_query about the transcript.

    Accepts:
      - metadata["audio_path"]: local file path to audio file
      - metadata["audio_b64"]: base64-encoded audio (written to temp file)
    """
    try:
        import whisper  # lazy import — model is large
    except ImportError:
        return (
            "Whisper is not installed. Run: pip install openai-whisper",
            {"error": "whisper_not_installed"},
        )

    audio_path: str | None = metadata.get("audio_path")

    # Write base64 audio to temp file if needed
    temp_file = None
    if not audio_path and metadata.get("audio_b64"):
        try:
            audio_bytes = base64.b64decode(metadata["audio_b64"])
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            audio_path = tmp.name
            temp_file = tmp.name
        except Exception as exc:  # noqa: BLE001
            return f"Failed to decode audio: {exc}", {"error": str(exc)}

    if not audio_path or not Path(audio_path).exists():
        return "Audio file not found.", {"error": "no_audio_file"}

    try:
        # Use "base" model — balance of speed and accuracy
        # TODO(phase-5): make model configurable via cfg
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        transcript: str = result.get("text", "").strip()
        logger.info("Whisper transcribed %d chars", len(transcript))
    except Exception as exc:  # noqa: BLE001
        logger.error("Whisper transcription failed: %s", exc)
        return f"Transcription failed: {exc}", {"error": str(exc)}
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    if not transcript:
        return "The audio appears to contain no speech.", {"transcript": ""}

    # Now answer the user's question about the transcript
    system_prompt = build_system_prompt(_AUDIO_SYSTEM, memory_context)
    user_prompt = (
        f"Audio transcript:\n{transcript}\n\n"
        f"User question: {user_query}"
    )
    try:
        answer = gateway.complete(
            role=ModelRole.GENERAL_SYNTHESIS,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        ).text.strip()
        return answer, {"transcript": transcript[:500], "audio_model": "whisper-base"}
    except Exception as exc:
        return (
            f"Transcript:\n{transcript}\n\n(Could not synthesize answer: {exc})",
            {"transcript": transcript[:500], "error": str(exc)},
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Example: test with a public image URL
    from mao.core.state import make_initial_state
    state = make_initial_state("What is in this image?", "user-test")
    state["metadata"] = {
        "modality": "image",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    }
    state = multimodal_node(state)
    print(state["response"])
