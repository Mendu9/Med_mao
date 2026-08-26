"""Routing and query-shaping prompts."""
from __future__ import annotations

from mao.prompts.registry import PromptSpec

ROUTER_CLASSIFY = PromptSpec(
    name="router.classify",
    version="1.0.0",
    output_contract="one intent label from ALL_INTENTS",
    required_variables=(),
    description="Classifies a user query into exactly one intent label.",
    template="""\
You are an intent classification router for a multi-agent AI system.
Classify the user query into EXACTLY ONE of these intent labels:

  summarize   - User wants a summary of a document or topic they provide
  graphrag    - User asks a factual/knowledge/science question (who, what, where, when, why,
                how does X work, explain X, what is X, what causes X, what are the symptoms of X)
                - including biomedical science questions about proteins, genes, mechanisms,
                pathways, disease biology, neuropathology, stroke, cardiovascular, dementia
  tool        - User needs live web search, a calculator, or a Wikipedia lookup
  multimodal  - User provides or asks about an image, audio, or non-text media
  critic      - User wants feedback, review, or evaluation of a text or plan
  clinical    - User provides an MRI scan, brain image, or medical report FOR ANALYSIS;
                asks about a SPECIFIC PATIENT'S scan results, Alzheimer's stage prediction
                for a patient, clinical decision support for brain imaging, or wants a
                structured medical report card generated from patient data
  chitchat    - Greetings, pleasantries, acknowledgements, off-topic conversation
                (hi, hello, thanks, yes, no, ok)
  fallback    - Query does not fit any above category

Rules:
  - Respond with ONLY the label word, nothing else.
  - When unsure between graphrag and tool, prefer graphrag.
  - Use clinical ONLY when the user is asking about a specific patient case, medical image,
    or report - NOT for general biomedical science questions (those are graphrag).
  - Examples of graphrag (NOT clinical): "what are tau tangles?", "explain amyloid cascade",
    "what does APOE4 do?", "how does neuroinflammation work?",
    "what causes brain stroke?", "what is ischemic stroke?", "what are stroke risk factors?",
    "how does dementia progress?", "what is the blood-brain barrier?"
  - Examples of clinical (NOT graphrag): "analyse this MRI", "what stage is this patient?",
    "summarise this medical report", "does this scan show Alzheimer's?"
  - When unsure between graphrag and summarize, check if the user provides
    a passage to summarize (summarize) or just asks a question (graphrag).
""",
)

ROUTER_USER = PromptSpec(
    name="router.user_turn",
    version="1.0.0",
    output_contract="text",
    required_variables=("memory_context", "history", "query"),
    description="User-side turn supplied to the router classifier.",
    template="""\
{memory_context}

Chat history (last 3 turns):
{history}

User query: {query}

Intent label:""",
)

DECOMPOSER_SPLIT = PromptSpec(
    name="decomposer.split",
    version="1.0.0",
    output_contract='JSON: {"sub_queries": [str, ...]}',
    required_variables=(),
    description="Splits a genuinely multi-part question into sub-questions.",
    template=(
        "You split complex biomedical questions into independent sub-questions. "
        "Return JSON only: {\"sub_queries\": [\"...\"]}. "
        "If the question is already single-part, return it unchanged as the only element."
    ),
)

DOMAIN_CLASSIFY = PromptSpec(
    name="domain.classify",
    version="1.0.0",
    output_contract="one domain label",
    required_variables=(),
    description="Assigns a biomedical domain used for retrieval filtering.",
    template=(
        "Classify the biomedical domain of the query. "
        "Answer with exactly one lowercase label: alzheimer, stroke, or general."
    ),
)

PROMPTS = (ROUTER_CLASSIFY, ROUTER_USER, DECOMPOSER_SPLIT, DOMAIN_CLASSIFY)
