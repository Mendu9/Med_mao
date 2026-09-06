r"""A hold-out corpus the implementation's vocabulary has never seen.

## Why this exists

`tests/core/pdf_layout_generator.py` stopped letting the implementer choose the
LAYOUTS, and that was the right move — but it left the implementer choosing the
WORDS, and an adversarial review walked straight through the gap. Its own
docstring says so:

    TRAILING_CLINICAL — "Chosen to END in a clinical head noun, which is the
    signal the boundary rule relies on."

A corpus selected to satisfy the mechanism under test measures the mechanism
against itself. Every phrase in that corpus ends in a word `CLINICAL_HEADS`
contains, so `has_clinical_head` always fires, so the boundary is always found,
so the suite is green — and 49 of 60 phrases drawn from *outside* that selection
were destroyed with no refusal at all.

So this module is built the other way round. The phrases and names here are
written first, from the domain, and then `test_the_corpus_is_genuinely_held_out`
asserts that a large fraction of them are **absent** from the implementation's
lexicon. That assertion is the point of the file:

  - it fails if the corpus drifts towards the vocabulary, and
  - it fails if somebody closes a hold-out failure by pasting these words into
    `lexicon.py`, which is the "open-ended vocabulary/regex patch cycle" the
    control documents forbid.

Nothing here imports the lexicon to BUILD anything. The lexicon is imported in
one place, by the meta-test, purely to prove independence.

## Combined axes

Each of ADV15-4 (a decorated value), ADV15-5 (a lexicon word inside a name) and
ADV15-6 (an invisible character inside a token) was found by varying one axis at
a time. `combined_documents()` varies them together, because a guard that only
ever sees one deviation at a time cannot see an interaction — and the phase
brief asks for "per-interaction/combined-axis coverage, not only one-axis-at-a-
time coverage".
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

# ---------------------------------------------------------------------------
# Identifiers. Synthetic, but shaped exactly like the real thing.
# ---------------------------------------------------------------------------

#: Surnames that are ALSO clinical eponyms. A dementia letter is full of them,
#: which is precisely why a clinical stop-list cannot own the name boundary:
#: `Parkinson`, `Pick`, `Down`, `Rankin` and `Braak` are all real British
#: surnames and all real neurology vocabulary.
EPONYM_SURNAMES: tuple[str, ...] = (
    "Parkinson", "Pick", "Down", "Rankin", "Braak", "Charcot", "Huntington",
    "Barthel", "Boston", "Montreal",
)

#: Ordinary English words that are also ordinary British middle names. Every one
#: of these is a real surname; none of them was chosen by an attacker.
COLLIDING_MIDDLE_NAMES: tuple[str, ...] = (
    "May", "Day", "Down", "Pick", "Note", "Fair", "New", "Sharp", "Grade",
    "Stage", "Level", "Range", "Score", "Result", "Total", "Present", "Care",
    "Block",
)

#: Given names and surnames across the naming systems a UK memory clinic sees.
#: Deliberately NOT the generator's list: an overlap would re-measure whatever
#: the generator already measures.
GIVEN_NAMES: tuple[str, ...] = (
    "Sarah", "Gordon", "Esme", "Ifeoma", "Rukhsana", "Aleksandr", "Mei",
    "Tobias",
)
SURNAMES: tuple[str, ...] = (
    "Okonkwo", "Nkemdirim", "Fairhurst", "Whitfield", "Ravensworth",
    "Bergqvist", "Lindqvist", "Achterberg",
)

#: Person labels, including the ambiguous ones (`Carer`, `Patient`) whose own
#: word is ordinary clinical English.
PERSON_LABELS: tuple[str, ...] = (
    "Patient Name", "Name", "Next of Kin", "Consultant", "Carer", "Surname",
)

#: (label, value) for the non-person identifier types, so a document can be a
#: credible patient banner without the NAME field being its own evidence.
OTHER_IDENTIFIERS: tuple[tuple[str, str, str], ...] = (
    ("MRN", "MRN", "RGT/44219/B"),
    ("MRN", "Hospital Number", "LDS9931C"),
    ("NHS", "NHS Number", "943 476 5919"),
    ("DOB", "Date of Birth", "12/03/1948"),
    ("PHONE", "Telephone", "0113 496 0231"),
    ("POSTCODE", "Postcode", "LS9 7TF"),
)

# ---------------------------------------------------------------------------
# Clinical vocabulary the implementation does not know
# ---------------------------------------------------------------------------

#: Titlecase clinical noun phrases: the exact shape of a person's name.
#:
#: Written from the domain — validated instruments, neuroanatomy, care-plan
#: terminology, ward and service names, implanted devices, trial and programme
#: names — and NOT filtered through `lexicon`. Several are cardiac
#: contraindications to the drug this system advises on, so losing one is a
#: patient-safety event rather than a cosmetic defect.
#:
#: None of these ends in a `CLINICAL_HEADS` word, which is the signal the
#: current boundary rule depends on. That is deliberate and is what makes the
#: corpus a hold-out rather than a second copy of the fixture set.
HOLDOUT_CLINICAL: tuple[str, ...] = (
    # instruments
    "Rockwood Frailty",
    "Zarit Burden Interview",
    "Wessex Head Injury Matrix",
    "Katz Independence Ladder",
    "Bayer Activities Instrument",
    "Nottingham Extended ADL",
    # neuroanatomy
    "Sylvian Fissure",
    "Corpus Callosum",
    "Substantia Nigra",
    "Nucleus Basalis",
    "Locus Coeruleus",
    "Fornix Column",
    "Cingulate Gyrus",
    "Insular Ribbon",
    # care plan
    "Continuing Healthcare Funding",
    "Best Interests Meeting",
    "Enhanced Recovery Pathway",
    "Escalation Ceiling Agreed",
    "Virtual Frailty Round",
    # ward and service
    "Ashworth Rehabilitation Unit",
    "Nightingale Bay Four",
    "Cedar Suite Annexe",
    "Beeches Intermediate Bay",
    "Ellerslie Outreach Service",
    "Thornbury Liaison Team",
    # implanted devices
    "Medtronic Azure Pacemaker",
    "Abbott Assurity Lead",
    "Biotronik Edora Device",
    "Reveal Linq Recorder",
    "Optivol Fluid Monitor",
    "Attain Performa Lead",
    "Sorin Kora Pacemaker",
    # trials and programmes
    "Aspree Extension Trial",
    "Sprint Mind Substudy",
    "Hermes Pooled Analysis",
    "Predict Cohort Enrolment",
    "Ideal Registry Entry",
    "Improving Access Programme",
)

# ---------------------------------------------------------------------------
# Deviations. Each of these was one finding's whole axis.
# ---------------------------------------------------------------------------

#: Characters that may sit between a label's separator and its value. None is
#: exotic: markdown emphasis, a typographic quote pypdf emits verbatim, a bullet
#: from a pasted list, a bracket around a record number.
VALUE_DECORATIONS: tuple[str, ...] = (
    "", "**", "*", '"', "'", "«", "“", "(", "•", "~", "_", "`",
    "/", "+", "&", "@", "%", "█",
)

#: Every placeholder this system emits, used as a caller-supplied decoy. A
#: marker in the INPUT can never establish that a span was produced by THIS run,
#: so none of these may change what is redacted.
FORGED_PLACEHOLDERS: tuple[str, ...] = (
    "[NAME]", "[MRN]", "[NHS]", "[DOB]", "[PHONE]", "[EMAIL]", "[POSTCODE]",
    "[ADDRESS]", "[ACCOUNT]", "[NI_NUMBER]", "[name]", "[[NAME]]",
    "[NAME][NAME]", "［NAME］",
)

#: Characters that occupy no visible advance width but are not Unicode `Cf`.
#:
#: The `Cf` category was the previous fix and it was the right INSTINCT applied
#: to the wrong property: the property wanted is "carries no visible content",
#: and `Cf` is a proper subset of it. These are the remainder, by category:
#: `Mn` combining marks, `Lo` Hangul fillers, `So` Braille blank, `Cc` controls.
INVISIBLE_CHARACTERS: dict[str, str] = {
    "U+034F COMBINING GRAPHEME JOINER": "͏",
    "U+17B4 KHMER VOWEL INHERENT AQ": "឴",
    "U+17B5 KHMER VOWEL INHERENT AA": "឵",
    "U+115F HANGUL CHOSEONG FILLER": "ᅟ",
    "U+1160 HANGUL JUNGSEONG FILLER": "ᅠ",
    "U+3164 HANGUL FILLER": "ㅤ",
    "U+FFA0 HALFWIDTH HANGUL FILLER": "ﾠ",
    "U+2800 BRAILLE PATTERN BLANK": "⠀",
    "U+0001 START OF HEADING": "",
    "U+0007 BELL": "",
    "U+000E SHIFT OUT": "",
    "U+001F UNIT SEPARATOR": "",
}

#: Characters already handled at the frozen SHA. They are kept as CONTROLS: a
#: change that closes the list above must not reopen these.
HANDLED_INVISIBLES: dict[str, str] = {
    "U+200B ZERO WIDTH SPACE": "​",
    "U+200D ZERO WIDTH JOINER": "‍",
    "U+00AD SOFT HYPHEN": "­",
    "U+FE0F VARIATION SELECTOR-16": "️",
    "U+2060 WORD JOINER": "⁠",
    "U+180E MONGOLIAN VOWEL SEPARATOR": "᠎",
}

#: What follows a person value on its own line. `second_label` and
#: `clinical_after` are the two forms the terminal-token fix does not cover.
TRAILING_FORMS: tuple[str, ...] = (
    "none", "sentence", "second_label", "clinical_after", "parenthetical",
)

#: How a label and its value are arranged. `orphan_above` and `orphan_block` are
#: the two-column extractions where the association is cross-line.
LAYOUTS: tuple[str, ...] = (
    "inline", "orphan_above", "orphan_block", "value_before_label",
)


@dataclass(frozen=True)
class Case:
    """One hold-out document plus the ground truth about it.

    `must_not_survive` and `must_survive` are asserted separately, because they
    are the two directions of the invariant and a fix for one has reopened the
    other five times in this project's history.
    """

    id: str
    text: str
    #: Identifiers that must not be READABLE in the scrubbed output.
    #:
    #: Compared against the output with zero-width characters removed, because
    #: that is what a reader — or a third-party model — actually sees. Asserting
    #: on the raw string instead would call `[NAME]<U+034F>dirim Okonkwo` a pass
    #: on the ground that `Nkemdirim` is not a literal substring of it, which is
    #: the exact failure this corpus exists to catch.
    must_not_survive: tuple[str, ...]
    #: Clinical content that must appear verbatim in the scrubbed output.
    must_survive: tuple[str, ...]
    #: Which axis or axes this case varies. Used for coverage accounting.
    axes: tuple[str, ...]


def _banner(label: str, value: str, decoration: str = "") -> str:
    return f"{label}: {decoration}{value}{_closing(decoration)}"


def _closing(decoration: str) -> str:
    """The matching closer, so a decorated value looks like real typography."""
    return {
        "**": "**", "*": "*", '"': '"', "'": "'", "«": "»",
        "“": "”", "(": ")", "_": "_", "`": "`",
    }.get(decoration, "")


def _trailing(form: str, position: int) -> str:
    if form == "sentence":
        return ". Please review at the next appointment."
    if form == "second_label":
        return "    MRN: RGT/44219/B"
    if form == "clinical_after":
        return "    " + HOLDOUT_CLINICAL[position % len(HOLDOUT_CLINICAL)]
    if form == "parenthetical":
        return " (deceased)"
    return ""


def _trailing_truth(form: str, position: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(must_not_survive, must_survive) contributed by the trailing form."""
    if form == "second_label":
        return ("RGT/44219/B",), ()
    if form == "clinical_after":
        return (), (HOLDOUT_CLINICAL[position % len(HOLDOUT_CLINICAL)],)
    return (), ()


# ---------------------------------------------------------------------------
# Case builders. Each returns documents that vary ONE named axis, and
# `combined_documents` varies several at once.
# ---------------------------------------------------------------------------

def name_boundary_documents() -> list[Case]:
    """A full name under a person label, with content after it on the line.

    The invariant is not "the whole line is redacted". It is that **no fragment
    of the name is left legible beside a placeholder claiming it was removed**,
    and that clinical content after the name is not eaten. Both directions on
    one document, which is what makes this the hard case.
    """
    cases: list[Case] = []
    for position, (surname, given, label, form) in enumerate(
        product(EPONYM_SURNAMES, GIVEN_NAMES[:4], PERSON_LABELS[:4], TRAILING_FORMS)
    ):
        tail = _trailing(form, position)
        gone, kept = _trailing_truth(form, position)
        cases.append(
            Case(
                id=f"name_boundary-{surname}-{given}-{label}-{form}".replace(" ", "_"),
                text=f"{label}: {given} {surname}{tail}\n",
                must_not_survive=(given, surname, *gone),
                must_survive=kept,
                axes=("name_boundary", f"trailing_{form}"),
            )
        )
    return cases


def middle_name_documents() -> list[Case]:
    """An ordinary English word used as a middle name.

    `Sarah May Okonkwo` is an entirely ordinary name and nobody chose it
    adversarially. A run that stops at `May` emits a placeholder over `Sarah`
    and prints the surname next to it.
    """
    return [
        Case(
            id=f"middle_name-{middle}-{surname}-{label}".replace(" ", "_"),
            text=f"{label}: {given} {middle} {surname}\n",
            must_not_survive=(given, middle, surname),
            must_survive=(),
            axes=("middle_name",),
        )
        for middle, surname, label, given in product(
            COLLIDING_MIDDLE_NAMES, SURNAMES[:3], PERSON_LABELS[:3], GIVEN_NAMES[:1]
        )
    ]


def decorated_value_documents() -> list[Case]:
    """A labelled value that does not begin at the first non-separator character.

    Markdown emphasis, a quotation mark, a bullet or a bracket is enough. No
    adversary is required; pypdf emits typographic quotes for a PDF that uses
    them, and a clinician pasting from a formatted note produces the rest.
    """
    cases: list[Case] = []
    for decoration in VALUE_DECORATIONS:
        cases.append(
            Case(
                id=f"decorated-NAME-{decoration or 'control'!r}",
                text=_banner("Patient Name", "Gordon Whitfield", decoration) + "\n",
                must_not_survive=("Gordon", "Whitfield"),
                must_survive=(),
                axes=("decorated_value",),
            )
        )
        for kind, label, value in OTHER_IDENTIFIERS:
            cases.append(
                Case(
                    id=f"decorated-{kind}-{label}-{decoration or 'control'!r}".replace(" ", "_"),
                    text=_banner(label, value, decoration) + "\n",
                    must_not_survive=(value,),
                    must_survive=(),
                    axes=("decorated_value",),
                )
            )
    return cases


def forged_placeholder_documents() -> list[Case]:
    """A caller-supplied placeholder must never suppress a real redaction.

    Four insertion shapes, because the append form was closed once and the
    insert form then opened. A marker in caller-controlled text cannot establish
    "this span was produced by this run" in ANY position, so the fix cannot be
    positional either.
    """
    cases: list[Case] = []
    for marker in FORGED_PLACEHOLDERS:
        shapes = {
            "appended": f"Patient Name: Esme Fairhurst {marker}\nMRN: 4451209 {marker}\n",
            "prefixed": f"Patient Name: {marker} Esme Fairhurst\nMRN: {marker} 4451209\n",
            "line_leading": f"{marker} Patient Name: Esme Fairhurst\n{marker} MRN: 4451209\n",
            "inserted_line": f"Patient Name:\n{marker}\nEsme Fairhurst\nMRN:\n{marker}\n4451209\n",
            "inline_satisfied": f"Patient Name: {marker}\nEsme Fairhurst\nMRN: {marker}\n4451209\n",
        }
        for shape, text in shapes.items():
            cases.append(
                Case(
                    id=f"forged-{shape}-{marker}",
                    text=text,
                    must_not_survive=("Esme", "Fairhurst", "4451209"),
                    must_survive=(),
                    axes=("forged_placeholder",),
                )
            )
    return cases


def invisible_character_documents() -> list[Case]:
    """A zero-width character inside an identifier token.

    Two failure directions on one input: the identifier survives whole, or a
    placeholder is emitted over the fragment before the character and the rest
    stays legible. The second is worse, and it is what a category denylist keeps
    producing.
    """
    templates: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("NAME", "Patient Name: Nkem{c}dirim Okonkwo", ("Nkemdirim", "Okonkwo")),
        ("MRN", "MRN: 44{c}51209", ("4451209",)),
        ("NHS", "NHS Number: 943 4{c}76 5919", ("943 476 5919",)),
        ("PHONE", "Telephone: 0113 4{c}96 0231", ("0113 496 0231",)),
        ("DOB", "Date of Birth: 12/0{c}3/1948", ("12/03/1948",)),
        ("POSTCODE", "Postcode: SW1A{c} 1AA", ("SW1A 1AA",)),
        ("EMAIL", "Email: gordon{c}.whitfield@example-nhs.uk",
         ("gordon.whitfield@example-nhs.uk",)),
    )
    cases: list[Case] = []
    everything = {**INVISIBLE_CHARACTERS, **HANDLED_INVISIBLES}
    for name, character in everything.items():
        for kind, template, residues in templates:
            cases.append(
                Case(
                    id=f"invisible-{kind}-{name.split()[0]}",
                    text=template.format(c=character) + "\n",
                    must_not_survive=residues,
                    must_survive=(),
                    axes=("invisible_character",),
                )
            )
    return cases


def clinical_preservation_documents() -> list[Case]:
    """A clinical phrase in every position a person label can reach it from.

    The four layouts are the ones a two-column extraction really produces. In
    each, the phrase is NOT a patient name, and the document says so by carrying
    a real identifier of another type in the same banner.
    """
    cases: list[Case] = []
    for position, (phrase, layout) in enumerate(product(HOLDOUT_CLINICAL, LAYOUTS)):
        label = PERSON_LABELS[position % len(PERSON_LABELS)]
        if layout == "inline":
            text = f"{label}: Gordon Whitfield\nMRN: RGT/44219/B\n{phrase}\n"
        elif layout == "orphan_above":
            text = f"{label}:\nMRN:\n{phrase}\nRGT/44219/B\n"
        elif layout == "orphan_block":
            text = f"{label}:\nMRN:\nDate of Birth:\n{phrase}\nRGT/44219/B\n12/03/1948\n"
        else:
            text = f"Gordon Whitfield | {label}\nRGT/44219/B | MRN\n{phrase}\n"
        cases.append(
            Case(
                id=f"clinical-{phrase}-{layout}".replace(" ", "_"),
                text=text,
                must_not_survive=("RGT/44219/B",),
                must_survive=(phrase,),
                axes=("clinical_preservation", f"layout_{layout}"),
            )
        )
    return cases


def combined_documents() -> list[Case]:
    """Several deviations on ONE document.

    Each finding above was found by varying a single axis. A guard that only
    ever sees one deviation at a time cannot see an interaction, and an
    interaction is what a real hostile — or a real PDF — produces.
    """
    cases: list[Case] = []
    invisibles = list(INVISIBLE_CHARACTERS.items())
    for index, (decoration, form) in enumerate(
        product(VALUE_DECORATIONS[:8], TRAILING_FORMS)
    ):
        character_name, character = invisibles[index % len(invisibles)]
        middle = COLLIDING_MIDDLE_NAMES[index % len(COLLIDING_MIDDLE_NAMES)]
        surname = EPONYM_SURNAMES[index % len(EPONYM_SURNAMES)]
        given = f"Sar{character}ah"
        tail = _trailing(form, index)
        gone, kept = _trailing_truth(form, index)
        value = f"{given} {middle} {surname}"
        cases.append(
            Case(
                id=f"combined-{index}-{decoration or 'plain'!r}-{form}-{character_name.split()[0]}",
                text=_banner("Patient Name", value, decoration) + tail + "\n",
                # "Sarah", not "Sar{invisible}ah": the assertion is about what a
                # reader sees, and the invisible is removed from both sides.
                must_not_survive=("Sarah", middle, surname, *gone),
                must_survive=kept,
                axes=(
                    "combined", "decorated_value", "middle_name",
                    "invisible_character", f"trailing_{form}",
                ),
            )
        )
    return cases


def all_cases() -> list[Case]:
    """Every hold-out document, in a stable order."""
    return [
        case
        for builder in (
            name_boundary_documents,
            middle_name_documents,
            decorated_value_documents,
            forged_placeholder_documents,
            invisible_character_documents,
            clinical_preservation_documents,
            combined_documents,
        )
        for case in builder()
    ]
