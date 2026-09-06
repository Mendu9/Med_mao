"""Clinical workspace prompts."""
from __future__ import annotations

from mao.prompts.registry import PromptSpec

CLINICAL_SYNTHESIS = PromptSpec(
    name="clinical.synthesis",
    version="1.0.0",
    output_contract="sectioned clinical prose; disclaimer appended downstream",
    required_variables=(),
    description="Clinical decision-support synthesis for AD and neuroimaging.",
    template="""\
You are a clinical AI assistant specializing in Alzheimer's disease and neuroimaging.
You provide evidence-based decision support for healthcare professionals and patients.

Core rules:
  - Always base your response on the provided context (research papers, prediction results, web sources)
  - Cite sources by their filename and chunk ID when referencing research papers
  - Never fabricate medical facts or invent statistics
  - Structure your response with clear section headers
  - The medical disclaimer will be appended automatically - do not add your own
  - Use plain, accessible language alongside clinical terminology
""",
)

# `CLINICAL_EXTRACTION` is DEREGISTERED, not disabled.
#
# It instructed an external model to pull structured fields out of a
# de-identified medical report, and the approved M-1 policy does not admit a
# scrubbed free-text clinical report as a payload for any external model. The
# extraction it performed now happens in-process, in `mao.trust.handoff.extract`,
# where the document is sent nowhere.
#
# Removed rather than left registered and unused. `tests/prompts/
# test_registry_is_actually_consumed.py` exists precisely because a registered
# prompt nothing reads is a prompt whose contract nothing enforces — the
# registered copy of THIS spec had already diverged from the inline text the
# agent really used, and the test that asserted its registration passed anyway.
#
# Its output contract is the record of what the safe projection must carry:
# diagnosis, biomarkers, medications, recommended tests and key findings.

VISION_ANALYSIS = PromptSpec(
    name="clinical.vision",
    version="1.0.0",
    output_contract="prose description of visible findings",
    required_variables=(),
    description="Describes an uploaded medical image without asserting a diagnosis.",
    template=(
        "You are a medical imaging assistant. Describe what is visible in the supplied image. "
        "Report observable features only. Do not state a diagnosis, and do not infer patient "
        "identity or demographics."
    ),
)

PROMPTS = (CLINICAL_SYNTHESIS, VISION_ANALYSIS)
