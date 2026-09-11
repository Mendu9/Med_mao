r"""One authorised entry per non-model external sink.

## The defect this closes

`authorise()` had three call sites at `b63311d`, all `MODEL_PROVIDER`, all in
`mao/providers/gateway.py`. The policy table declared seven destinations. Web
search, the scholarly API, the vector store and external memory reached the
network with no `authorise()` call at all, so for those four sinks the policy
row was documentation: the `(EXTERNAL_MEMORY, MEMORY_WRITE)` row and the three
`EVIDENCE_SEARCH` rows were rows nothing consulted.

The table lookup is not the part that matters most. `authorise()` is also where
the run-scoped `RequestProtection.leaked_in` assertion runs - the control
`gateway` describes as "the one that closes the class of defect the previous six
waves kept reopening" - and on the four bypassing paths that control did not
exist. A de-identification defect that put a patient's name into a retrieval
query was refused on its way to the synthesis model and sent to the web-search
provider and to NCBI without comment.

## Why the wrapper lives at the SINK and not at the agent

`_web_search_clinical` is one caller of `mao.core.web_search.web_search`. An
`authorise()` in that agent would protect that one path and leave the next
caller unguarded, which is the "missing call site on a second channel" shape
this whole phase exists to end - and it is the shape the audio channel, the
history channel and the ragas client each took in turn. So each wrapper below
is called from the module that makes the network call, and the agents call
nothing new.

## Why every wrapper declares its class explicitly

There is no default. `complete()` had one, `multimodal_agent` said nothing, and
the signature declared `SAFE_DERIVED_TEXT` on behalf of a base64 patient scan.
A trust class a call site did not state is not a declaration.
"""
from __future__ import annotations

from collections.abc import Sequence

from mao.trust.classes import TrustClass
from mao.trust.egress.gateway import authorise
from mao.trust.egress.policy import Destination, EgressPurpose


def _authorise(
    destination: Destination, purpose: EgressPurpose, texts: Sequence[str]
) -> None:
    """Authorise one non-model call, or raise `EgressRefused`.

    The envelope is discarded deliberately. These sinks take a plain string and
    have nowhere to carry an `ExternalSafePayload`; what is wanted here is the
    DECISION - the flow is named, the class is admitted, and no identifier this
    request removed is in the outgoing text - not the envelope. Phase 2 owes
    these routes the typed `SafeEvidenceQuery` the policy row already admits.
    """
    authorise(
        destination=destination,
        purpose=purpose,
        trust_class=TrustClass.SAFE_DERIVED_TEXT,
        texts=tuple(text for text in texts if text),
    )


def authorise_web_search(query: str) -> None:
    """The multi-provider fallback chain in `mao.core.web_search`."""
    _authorise(Destination.WEB_SEARCH, EgressPurpose.EVIDENCE_SEARCH, [query])


def authorise_scholarly(query: str) -> None:
    """Live PubMed/NCBI search."""
    _authorise(Destination.SCHOLARLY_API, EgressPurpose.EVIDENCE_SEARCH, [query])


def authorise_vector_query(texts: Sequence[str]) -> None:
    """A chroma query.

    Authorised unconditionally rather than only when `chroma_host` names a
    remote. A local store is not an external sink, but the call site cannot
    tell the difference and a deployment that turns on a remote host must not
    silently become an unguarded egress. The identifier assertion costs nothing
    when the store is local.
    """
    _authorise(Destination.VECTOR_STORE, EgressPurpose.EVIDENCE_SEARCH, texts)


def authorise_vector_embedding(texts: Sequence[str]) -> None:
    """Text handed to an embedding model on the way into the vector store."""
    _authorise(Destination.VECTOR_STORE, EgressPurpose.EMBEDDING, texts)


def authorise_memory_write(texts: Sequence[str]) -> None:
    """A write into external/general memory.

    `ADV15-9`/`G-8c` is closed - the content written is protected text - but the
    architecture review carried a caveat into A-4: the write was not routed
    through `authorise()`, so the policy row was never consulted and the
    identifier assertion never ran on this sink. This closes the caveat.
    """
    _authorise(Destination.EXTERNAL_MEMORY, EgressPurpose.MEMORY_WRITE, texts)


def authorise_memory_read(texts: Sequence[str]) -> None:
    """A search against external/general memory.

    A read sends this request's query to the same third party the write sends
    its content to. There was no policy row for it at `b63311d`, so this call
    could not have been authorised even by a call site that wanted to be.
    """
    _authorise(Destination.EXTERNAL_MEMORY, EgressPurpose.MEMORY_READ, texts)
