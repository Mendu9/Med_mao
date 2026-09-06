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
from mao.trust.classes import InputChannel
from mao.trust.egress.policy import EgressPurpose
from mao.trust.inputs import limits
from mao.trust.inputs.boundary import protect_channel
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
    r"""Transcribe audio with Whisper, protect the transcript, then answer.

    ## ADV15-15 — the modality that never de-identified anything

    `grep -n "scrub_pii\|find_ambiguities" mao/agents/multimodal_agent.py`
    returned nothing. This function transcribed the upload and concatenated the
    raw Whisper output straight into the user turn, and `result_meta` returned
    500 characters of it to the client, where it was cached in Redis and written
    to a trace. A recorded consultation contains spoken names, addresses and
    dates of birth by construction, and `audio_b64` is the documented way to
    send one.

    It was never a de-identification defect: the de-identifier was not called.
    `01_ARCHITECTURE.md` — "any existing audio/transcript path must obey the
    same protected-input contract or be disabled" — so the transcript now enters
    the same boundary as typed text, on the `TRANSCRIPT` channel, with the
    identifiers it removes recorded against THIS request's protection so the
    egress assertion covers them exactly as it covers the query.

    A spoken patient header is as unresolvable as a written one, and the
    recording is processed unseen for the same reason a PDF is, so
    `protect_channel` refuses an ambiguous one and the route answers 422 asking
    for structured fields. That refusal must reach the route, which is why
    nothing here catches it.

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
        except Exception as exc:  # noqa: BLE001
            return f"Failed to decode audio: {exc}", {"error": str(exc)}

        # ADV15-7 on this channel. The report path bounds its decoded
        # attachment; audio had no bound at all, so a caller could hand the
        # transcriber an arbitrarily large recording and hold a worker thread
        # for as long as it takes to transcribe it. Same shared cap, refused
        # rather than truncated — a half-transcribed consultation answered as
        # though it were complete is the failure the report path refuses too.
        # Raised outside the broad `except` above deliberately: swallowed here
        # it would become an answer built on no audio at all.
        limits.check(
            "the decoded audio",
            len(audio_bytes),
            limits.MAX_DECODED_ATTACHMENT_BYTES,
        )

        try:
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
        return "The audio appears to contain no speech.", {"transcript_chars": 0}

    # The transcript is a sensitive input channel like any other. Bounded first
    # — speech is slower than typing, so an hour of dictation is well under this
    # and anything above it is not a consultation — then de-identified exactly
    # once, here, and never again.
    limits.check("the transcript", len(transcript), limits.MAX_TRANSCRIPT_CHARS)
    safe_transcript = protect_channel(transcript, InputChannel.TRANSCRIPT)

    # Now answer the user's question about the transcript
    system_prompt = build_system_prompt(get_prompt("multimodal.audio_transcript").template, memory_context)
    user_prompt = (
        f"Audio transcript:\n{safe_transcript.text}\n\n"
        f"User question: {user_query}"
    )
    # `result_meta` is echoed into the response body, cached in Redis under two
    # keys and written to a trace, so what goes in it is an egress decision like
    # any other. It used to carry `transcript[:500]` — the RAW Whisper output,
    # i.e. the first 500 characters of a consultation, which is precisely where
    # the spoken patient header is. Only the length is reported now; the
    # transcript itself is what the answer was built from and does not need
    # returning alongside it.
    result_meta = {"transcript_chars": len(safe_transcript.text), "audio_model": "whisper-base"}
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
        return answer, result_meta
    except Exception as exc:
        # The de-identified transcript, not the raw one. A provider outage
        # should not throw away a transcription the clinician is waiting for,
        # and this text is the same trust class as the de-identified query the
        # other routes already return and persist.
        return (
            f"Transcript:\n{safe_transcript.text}\n\n(Could not synthesize answer: {exc})",
            {**result_meta, "error": str(exc)},
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
