r"""How much a caller may put through the protected boundary.

## The finding

An unauthenticated 194 KB request pinned one of eight worker threads for 34
seconds, and doubling the page count roughly quadrupled the cost:

    pages   request body   extracted            extract  ambiguity  scrub    total
      10       13,224 B    32,518 c /  1,018 l   0.25 s    0.03 s   0.30 s   0.57 s
      40       49,236 B   130,078 c /  4,078 l   0.50 s    0.11 s   2.62 s   3.23 s
      80       97,432 B   260,158 c /  8,158 l   1.00 s    0.21 s   8.57 s   9.79 s
     160      194,108 B   520,318 c / 16,318 l   1.77 s    0.40 s  32.07 s  34.24 s

`ChatRequest.query` was capped at 8,000 characters and `metadata` was an
unvalidated `dict[str, Any]`, so the cap applied to the one channel that did not
need it. Fourteen requests a minute from a single IP — inside the 60/min rate
limit, which itself fails open when Redis is down — demand more CPU than the
pool has.

The remaining O(n^2) in `_pair_by_type` is a separate, deferred item. A cap is
not a substitute for fixing it; it is what makes the deferral safe, which is why
the review called a cap the minimum viable fix and why it is Phase 1 scope while
the algorithmic rewrite is not.

## Refuse, do not truncate

A truncated clinical report answered as though it were complete is the same
class of failure as a deleted clinical phrase: the model reasons over content
that is missing, and nothing downstream can tell. So an oversized report is
REFUSED with a 413 and the caller is told, rather than silently shortened.

Chat history is the exception, and deliberately: the agents already read only
the last three or four turns, so keeping the most recent turns discards nothing
any downstream consumer would have seen. Dropping the oldest turn of a long
conversation is not a silent deletion of clinical content — it is the behaviour
that already existed, now bounded before the expensive work rather than after.
"""
from __future__ import annotations


class InputTooLarge(Exception):
    """A caller sent more than the protected boundary will process.

    Carries the channel and the limit, never the content. This is raised on the
    path that handles patient data, its message is logged, and it reaches an
    HTTP body.
    """

    def __init__(self, channel: str, size: int, limit: int) -> None:
        super().__init__(
            f"{channel} is {size} units, above the {limit} this endpoint will "
            "process. Send a smaller document, or split it."
        )
        self.channel = channel
        self.size = size
        self.limit = limit


#: Decoded attachment bytes. Bounds base64 decoding and the PDF reader itself,
#: before any text has been extracted. Four megabytes is a large real report and
#: an order of magnitude below the size at which extraction alone becomes slow.
MAX_DECODED_ATTACHMENT_BYTES = 4_000_000

#: Extracted text characters, applied BEFORE `find_ambiguities` and `scrub_pii`.
#: This is the cap that matters: it is the input to the quadratic. 120,000
#: characters is roughly 37 pages and measured at about 2.5 seconds, which one
#: of eight workers can absorb; 520,000 characters took 32.
MAX_EXTRACTED_TEXT_CHARS = 120_000

#: A transcript is speech, and speech is slower than typing: an hour of dictation
#: is well under this.
MAX_TRANSCRIPT_CHARS = 60_000

#: The query cap `ChatRequest` already declares, restated here so the boundary
#: does not depend on a validator it does not own.
MAX_QUERY_CHARS = 8_000

#: Chat history. Turns beyond the most recent are dropped, not refused — see the
#: module docstring. A single oversized turn IS refused, because it is a
#: document pasted into a message and the report reasoning applies to it.
MAX_HISTORY_TURNS = 20
MAX_HISTORY_TURN_CHARS = 8_000

#: One structured patient field. A field longer than this is a document in a
#: field, which is the shape the structured pathway exists to avoid.
MAX_STRUCTURED_FIELD_CHARS = 2_000


def check(channel: str, size: int, limit: int) -> None:
    """Refuse `channel` if it is over `limit`. Never truncates."""
    if size > limit:
        raise InputTooLarge(channel, size, limit)
