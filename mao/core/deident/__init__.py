"""Deterministic de-identification.

Split from the single `pii_scrubber` module in Wave 11, because the invariant it
has to hold is really two invariants that pull in opposite directions and were
being enforced by one tangle of regexes:

    (a) no raw identifier survives to any external or persistent sink;
    (b) no clinical content is deleted or altered, and no placeholder is emitted
        unless a correctly typed identifier was actually removed.

`lexicon`  what words are never part of a person's name
`fields`   what field labels exist and what type of value each carries
`values`   what a value of a given type looks like, positively and boundedly
`layout`   where in a document each identifier is, and redaction in place
`freetext` identifiers with no field label at all, matched by shape
"""
