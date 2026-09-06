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


from mao.memory.mem0_handler import build_system_prompt
from mao.prompts import get_prompt
from mao.providers import gateway
from mao.trust.egress.policy import EgressPurpose
from mao.providers.registry import ModelRole

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# There is deliberately no `multimodal_node`.
#
# It was removed in Wave 7. It had not been a graph node since modality became a
# capability of `clinical_node`, and it was unreachable-with-media even before
# that: `router_node` forces EVERY attachment to `clinical`, so the node could
# only ever have been entered with nothing attached — in which case its only
# possible reply was "please provide an image or audio file". An agent that
# cannot receive the data it exists to process is not a design choice.
#
# It also carried the last copy of the echo-caller-metadata pattern that
# adversarial H-2 was about, and its own `ATTACHMENT_KEYS` filter, both of which
# would have had to be maintained in step with `clinical_node`'s.
#
# The CAPABILITIES survive and are what `clinical_node` calls: `handle_image`
# and `handle_audio` below. 00_RULES: "Do not create an 'agent' where a
# deterministic function/workflow/tool is sufficient."
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Image handler
# ---------------------------------------------------------------------------

def handle_image(  # public: shared with `clinical_agent`
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
        # Through the shared guard, never `requests.get` directly — the URL is
        # caller-supplied. See `mao/safety/fetch.py`.
        from mao.safety.fetch import fetch_image_bytes

        try:
            image_b64 = base64.b64encode(
                fetch_image_bytes(metadata["image_url"])
            ).decode("utf-8")
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
            purpose=EgressPurpose.IMAGE_ANALYSIS,
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
    system_prompt = build_system_prompt(get_prompt("multimodal.audio_transcript").template, memory_context)
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
            purpose=EgressPurpose.GENERAL_SYNTHESIS,
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
    # Exercises the capability directly, which is how `clinical_node` uses it.
    description, meta = handle_image(
        "What is in this image?",
        {
            "image_url": (
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/"
                "PNG_transparency_demonstration_1.png/280px-"
                "PNG_transparency_demonstration_1.png"
            ),
        },
        "",
    )
    print(description)
    print("meta:", meta)
