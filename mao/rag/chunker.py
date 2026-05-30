"""Adaptive semantic chunker for biomedical text.

Uses sentence-transformer cosine similarity to detect semantic boundaries,
achieving 87% accuracy in clinical RAG vs 50% for fixed-size chunking.
Reference: PMC12649634.
"""
from __future__ import annotations
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Section header patterns for structured clinical/research docs
_SECTION_PATTERNS = re.compile(
    r"^(?:Abstract|Introduction|Background|Methods?|Materials? and Methods?|"
    r"Results?|Discussion|Conclusion|References?|Acknowledgements?|"
    r"Case Report|Patient History|Assessment|Plan|HPI|"
    r"Medications?|Diagnosis|Treatment|Follow.?up)\s*[:.\n]",
    re.IGNORECASE | re.MULTILINE,
)


_embed_model = None  # module-level cache — avoids reloading 400 MB on every call


def _get_embed_model():
    """Lazy-load the sentence transformer model for similarity computation."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        from mao.core.config import cfg
        _embed_model = SentenceTransformer(cfg.embed_model)
        return _embed_model
    except Exception:
        return None


def adaptive_biomedical_chunk(
    text: str,
    similarity_threshold: float = 0.80,
    max_words: int = 500,
    min_words: int = 60,
    overlap_sentences: int = 2,
) -> list[str]:
    """Split text into semantically coherent chunks with sentence-level sliding overlap.

    Detects chunk boundaries where consecutive sentence embeddings drop below
    similarity_threshold, then carries the last `overlap_sentences` sentences
    of each chunk into the start of the next chunk to preserve cross-boundary
    clinical context (e.g. a diagnosis sentence followed by treatment detail).

    Falls back to fixed ~400-word splits if sentence-transformers unavailable.

    Args:
        text: Input document text.
        similarity_threshold: Cosine similarity below which a new chunk starts.
        max_words: Hard cap on chunk size (words).
        min_words: Minimum chunk size before a boundary is accepted.
        overlap_sentences: Number of trailing sentences to carry into the next chunk.

    Returns:
        List of chunk strings.
    """
    if not text or not text.strip():
        return []

    model = _get_embed_model()
    if model is None:
        logger.warning("sentence-transformers unavailable; using fixed-size chunking fallback")
        return _fixed_chunk(text, max_words)

    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            sentences = nltk.sent_tokenize(text)
    except ImportError:
        sentences = _split_sentences_simple(text)

    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    try:
        import numpy as np
        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_words = len(sentences[0].split())

        for i in range(1, len(sentences)):
            word_count = len(sentences[i].split())
            sim = float(
                np.dot(embeddings[i - 1], embeddings[i])
                / (np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8)
            )
            should_split = (
                (sim < similarity_threshold or current_words + word_count > max_words)
                and current_words >= min_words
            )
            if should_split:
                chunks.append(" ".join(current))
                # Carry the last `overlap_sentences` sentences into the next chunk
                # so cross-boundary context (e.g. diagnosis → treatment) is preserved.
                overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current = overlap + [sentences[i]]
                current_words = sum(len(s.split()) for s in current)
            else:
                current.append(sentences[i])
                current_words += word_count

        if current:
            chunks.append(" ".join(current))
        return [c for c in chunks if c.strip()]

    except Exception as exc:
        logger.warning("Adaptive chunking failed (%s); falling back to fixed-size", exc)
        return _fixed_chunk(text, max_words)


def section_aware_split(text: str) -> dict[str, str]:
    """Split structured documents (research papers, clinical notes) by section headers.

    Returns {section_name: section_text} preserving document structure.
    """
    sections: dict[str, str] = {}
    matches = list(_SECTION_PATTERNS.finditer(text))

    if not matches:
        return {"body": text}

    for idx, match in enumerate(matches):
        section_name = match.group().strip().rstrip(":.\n").strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[section_name] = content

    # Prepend any text before the first section as "preamble"
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections = {"preamble": preamble, **sections}

    return sections


def chunk_document(
    text: str,
    doc_id: str,
    metadata: dict[str, Any] | None = None,
    use_sections: bool = True,
    similarity_threshold: float = 0.80,
    max_words: int = 500,
    min_words: int = 60,
    overlap_sentences: int = 2,
) -> list[dict[str, Any]]:
    """Full pipeline: section-split -> adaptive-chunk -> attach metadata.

    Returns list of chunk dicts ready for ChromaDB upsert:
    {chunk_id, text, source, domain, section, chunk_index, word_count, ...metadata}
    """
    import hashlib

    meta = metadata or {}
    chunks_out: list[dict[str, Any]] = []
    chunk_index = 0

    if use_sections:
        sections = section_aware_split(text)
    else:
        sections = {"body": text}

    for section_name, section_text in sections.items():
        text_chunks = adaptive_biomedical_chunk(
            section_text,
            similarity_threshold=similarity_threshold,
            max_words=max_words,
            min_words=min_words,
            overlap_sentences=overlap_sentences,
        )
        for chunk_text in text_chunks:
            chunk_id = hashlib.md5(f"{doc_id}:{chunk_index}:{chunk_text[:50]}".encode()).hexdigest()
            chunk_record: dict[str, Any] = {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": doc_id,
                "section": section_name,
                "chunk_index": chunk_index,
                "word_count": len(chunk_text.split()),
                **meta,
            }
            chunks_out.append(chunk_record)
            chunk_index += 1

    return chunks_out


def _fixed_chunk(text: str, max_words: int = 400) -> list[str]:
    """Fallback: split into fixed-size word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i : i + max_words])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def _split_sentences_simple(text: str) -> list[str]:
    """Minimal sentence splitter when nltk is unavailable."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
