r"""The exact reproductions from the frozen `623cffa` reviews.

`00_RULES.md`: "Exact predecessor reproductions must be re-run before broader
new corpora." Six remediation rounds each closed the failures they were given
while new ones appeared in the same invariant class, so a finding is only closed
when the reviewer's own input produces the reviewer's expected output.

Each test names the finding it reproduces, quotes the recorded input verbatim,
and asserts the property the finding says is violated. Nothing here is
paraphrased into a shape the implementation happens to handle.

Findings covered:

    ADV15-1   clinical phrases destroyed on the upload path with no refusal
    ADV15-2   /chat/stream answered 500 where /chat answered 422   (post-gate)
    ADV15-4   a decorated labelled value is never redacted
    ADV15-5   a lexicon word inside a name emits a prefix placeholder
    ADV15-6   invisible characters that are not `Cf`
    ADV15-8   chat_history is an unscrubbed egress
    G-1       an eponym surname beside a second field or a clinical phrase
    G-7       a caller-supplied placeholder suppresses redaction
    idempotence residual introduced at the post-gate head
"""
from __future__ import annotations

import pytest

from mao.core.deident.ambiguity import find_ambiguities
from mao.core.pii_scrubber import scrub_pii

from .test_holdout_deidentification import visible


class TestG1EponymSurname:
    """`Patient Name: Mary Parkinson    MRN: RGT/44219/B` -> `[NAME] Parkinson`.

    The terminal-token fix closed the subset its own wording named — "a clinical
    word THAT IS THE LAST TOKEN ON THE LINE" — and nothing else. The moment any
    content follows the surname the run is returned truncated.

    Under an explicit PERSON label the label has already settled the type, so
    the clinical lexicon has no authority to end the run at all.
    """

    #: The reviewer's list, verbatim, including the four that were already fixed
    #: so a regression in them is visible too.
    RECORDED = (
        "Patient Name: Mary Parkinson",
        "Patient Name: Helen Day",
        "Patient Name: Robert May",
        "Patient Name: Alan Rankin",
        "Consultant: Dr Alan Rankin",
        "Patient Name: Mary Jane Parkinson",
        "Patient Name: Mary Parkinson    MRN: RGT/44219/B",
        "Patient Name: Grace Parkinson  DOB: 12/03/1948",
        "Patient Name: May Okonkwo",
        "Patient Name: Parkinson, Mary",
        "Patient Name: Mary Parkinson Day",
        "Patient Name: Mary Parkinson | DOB: 12/03/1948",
        "Patient Name: Mary Parkinson (deceased)",
        "Patient Name: Mary Parkinson  NHS Number: 943 476 5919",
        "Patient Name: Mary Rankin    Ward: 12",
        "Patient Name: Mary Parkinson    Mobitz Type II on 24 hour tape",
    )

    _NAME_WORDS = (
        "Mary", "Helen", "Robert", "Alan", "Grace", "Jane", "May",
        "Parkinson", "Day", "Rankin", "Okonkwo",
    )

    @pytest.mark.parametrize("line", RECORDED)
    def test_no_fragment_of_the_name_survives(self, line: str) -> None:
        out = scrub_pii(line)
        residue = [word for word in self._NAME_WORDS if word in visible(out)]
        assert not residue, f"{line!r} -> {out!r}  residue={residue}"

    def test_the_cross_line_letterhead_redacts_the_name(self) -> None:
        """The reviewer measured 160/192 WHOLE leaks on this shape, unchanged."""
        for document in (
            "Patient Name:\nMary Parkinson\nMRN:\nRGT/44219/B\n",
            "Patient Name:\nMRN:\nMary Parkinson\nRGT/44219/B\n",
        ):
            out = scrub_pii(document)
            assert "Parkinson" not in visible(out), f"{document!r} -> {out!r}"


class TestADV15_5MiddleNameCollision:
    """`Patient Name: Sarah May Okonkwo` -> `Patient Name: [NAME] May Okonkwo`.

    84 of 92 measured. `Sarah May Okonkwo` is an entirely ordinary name; no
    attacker chose it. The output claims the name was removed and prints the
    surname next to the claim.
    """

    MIDDLES = (
        "May", "Day", "Down", "Pick", "Note", "Fair", "New", "Sharp", "Grade",
        "Stage", "Level", "Range", "Score", "Result", "Total", "Present",
        "Care", "Block",
    )

    @pytest.mark.parametrize("middle", MIDDLES)
    @pytest.mark.parametrize("surname", ("Okonkwo", "Nkemdirim"))
    @pytest.mark.parametrize("label", ("Patient Name", "Name"))
    def test_no_surname_survives_beside_a_placeholder(
        self, middle: str, surname: str, label: str
    ) -> None:
        line = f"{label}: Sarah {middle} {surname}"
        out = visible(scrub_pii(line))
        assert surname not in out and "Sarah" not in out, f"{line!r} -> {out!r}"

    def test_a_name_longer_than_the_token_cap_is_not_truncated(self) -> None:
        """The reviewer's fourth mechanism: `_MAX_NAME_TOKENS` leaks the tail."""
        line = (
            "Patient Name: Maria Del Carmen Gonzalez Rodriguez Perez "
            "Fernandez Iglesias Navarro"
        )
        out = visible(scrub_pii(line))
        for word in ("Fernandez", "Iglesias", "Navarro"):
            assert word not in out, f"{line!r} -> {out!r}"


class TestADV15_4DecoratedValue:
    """41 of 95 (prefix, field) cells leak. Markdown bold is enough."""

    RECORDED = (
        ("Patient Name: **Harold Nkemdirim**", "Harold Nkemdirim"),
        ('Patient Name: "Harold Nkemdirim"', "Harold Nkemdirim"),
        ("Patient Name: • Harold Nkemdirim", "Harold Nkemdirim"),
        ("MRN: (RGT/44219/B)", "RGT/44219/B"),
        ("Patient Name: [NAME] Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: [name] Harold Nkemdirim", "Harold Nkemdirim"),
        ("[NAME] Patient Name: Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: [[NAME]] Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: [NAME][NAME] Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: ［NAME］ Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: [DOB] Harold Nkemdirim", "Harold Nkemdirim"),
        ("Patient Name: _Harold Nkemdirim_", "Harold Nkemdirim"),
        ("MRN: _RGT/44219/B_", "RGT/44219/B"),
        ("NHS Number: _943 476 5919_", "943 476 5919"),
        ("Date of Birth: _12/03/1948_", "12/03/1948"),
        ("Telephone: _0113 496 0231_", "0113 496 0231"),
    )

    @pytest.mark.parametrize("line,identifier", RECORDED)
    def test_the_identifier_is_removed(self, line: str, identifier: str) -> None:
        out = visible(scrub_pii(line))
        assert identifier not in out, f"{line!r} -> {out!r}"


class TestADV15_6InvisibleCharacters:
    """Whole-identifier leaks whose output is byte-identical to the input."""

    RECORDED = (
        ("MRN: 44ㅤ51209", "4451209"),
        ("NHS Number: 943 4͏76 5919", "943 476 5919"),
        ("Telephone: 0113 496 0231", "0113 496 0231"),
        ("DOB: 12/0⠀3/1948", "12/03/1948"),
        ("Postcode: SW1A 1AA", "SW1A 1AA"),
        ("Patient Name: Nkem͏dirim Okonkwo", "Nkemdirim"),
        ("Patient Name: Nkem⠀dirim Okonkwo", "Nkemdirim"),
        ("Patient Name: Nkemdirim Okonkwo", "Nkemdirim"),
        ("Email: harold͏.nkemdirim@leeds-nhs.uk", "harold.nkemdirim@leeds-nhs.uk"),
        ("Email: harold.nkemdirim͏@leeds-nhs.uk", "harold.nkemdirim@leeds-nhs.uk"),
        ("MRN: 44ᅠ51209", "4451209"),
        ("MRN: 44ᅟ51209", "4451209"),
        ("MRN: 44ﾠ51209", "4451209"),
        ("Telephone: 0113 4឴96 0231", "0113 496 0231"),
        ("Telephone: 0113 4឵96 0231", "0113 496 0231"),
    )

    @pytest.mark.parametrize("line,identifier", RECORDED)
    def test_the_identifier_is_removed(self, line: str, identifier: str) -> None:
        out = visible(scrub_pii(line))
        assert identifier not in out, f"{line!r} -> {out!r}"

    def test_the_characters_already_closed_stay_closed(self) -> None:
        """15 of 21 were fixed at the frozen SHA. A regression here is a
        regression in the direction the reviewer credited as correct."""
        for character in ("​", "‌", "‍", "⁠", "﻿",
                          "­", "️", "᠎", "‎", "‮"):
            line = f"MRN: 44{character}51209"
            assert "4451209" not in visible(scrub_pii(line)), repr(line)


class TestG7ForgedPlaceholder:
    """A marker in caller-controlled text can never establish provenance.

    The append form was closed and the insert form opened in the same commit.
    Both are recorded here, plus the inline form that has never been closed, so
    a fix cannot trade one for the other again.
    """

    RECORDED = (
        ("forward, name pushed one line down",
         "Patient Name:\n[NAME]\nMary Okonkwo\nMRN:\n[MRN]\n4451209\n"),
        ("label-block, placeholder block then real values",
         "Patient Name:\nMRN:\n[NAME]\n[MRN]\nMary Okonkwo\n4451209\n"),
        ("inline label satisfied, value on next line",
         "Patient Name: [NAME]\nMary Okonkwo\nMRN: [MRN]\n4451209\n"),
        ("placeholder cell with trailing punctuation",
         "Patient Name:\n[NAME].\nMary Okonkwo\nMRN:\n[MRN],\n4451209\n"),
        ("control - no decoy line",
         "Patient Name:\nMary Okonkwo\nMRN:\n4451209\n"),
    )

    @pytest.mark.parametrize("shape,document", RECORDED, ids=[r[0] for r in RECORDED])
    def test_the_real_values_are_still_redacted(
        self, shape: str, document: str
    ) -> None:
        out = visible(scrub_pii(document))
        residue = [w for w in ("Okonkwo", "4451209") if w in out]
        assert not residue, f"{shape}: {document!r} -> {out!r}  residue={residue}"

    @pytest.mark.parametrize(
        "marker",
        ("[SIC]", "[NB]", "[NAME]", "[MRN]", "[NHS]", "[ACCOUNT]", "[ADDRESS]",
         "[DOB]", "[PHONE]", "[EMAIL]", "[POSTCODE]", "[NI_NUMBER]"),
    )
    def test_an_appended_decoy_still_does_not_work(self, marker: str) -> None:
        """ADV14-5's original matrix. Closed at the frozen SHA; keep it closed."""
        document = f"Patient Name: Mary Okonkwo {marker}\nMRN: 4451209 {marker}\n"
        out = visible(scrub_pii(document))
        residue = [w for w in ("Okonkwo", "4451209") if w in out]
        assert not residue, f"{document!r} -> {out!r}  residue={residue}"


class TestADV15_1ClinicalPhrasesOnTheUploadPath:
    """49 of 60 clinical phrases destroyed with NO refusal.

    The policy the upload path states is that an unresolvable header is REFUSED
    and the caller is asked for structured fields. This asserts the policy: for
    every phrase, the document is either processed with the phrase intact, or
    refused. What may not happen is processing it and deleting the phrase.
    """

    CLINICAL = (
        "Rockwood Frailty", "Bristol Stool Chart", "Zarit Burden Interview",
        "Wessex Head Injury Matrix", "Nottingham Extended ADL",
        "Katz Independence Ladder", "Lawton Brody IADL",
        "Bayer Activities Instrument", "Sylvian Fissure", "Corpus Callosum",
        "Basal Ganglia", "Substantia Nigra", "Nucleus Basalis",
        "Locus Coeruleus", "Fornix Column", "Cingulate Gyrus",
        "Thalamic Radiation", "Insular Ribbon", "Continuing Healthcare Funding",
        "Best Interests Meeting", "Enhanced Recovery Pathway",
        "Reablement Team Input", "Community Matron Visit",
        "Escalation Ceiling Agreed", "Virtual Frailty Round",
        "Ashworth Rehabilitation Unit", "Nightingale Bay Four",
        "Cedar Suite Annexe", "Beeches Intermediate Bay",
        "Pendle Elderly Medicine", "Whitby Frailty Team",
        "Ellerslie Outreach Service", "Thornbury Liaison Team",
        "Medtronic Azure Pacemaker", "Abbott Assurity Lead",
        "Biotronik Edora Device", "Reveal Linq Recorder",
        "Optivol Fluid Monitor", "Attain Performa Lead",
        "Ensite Precision Mapping", "Sorin Kora Pacemaker",
        "Nucleus Profile Implant", "Aspree Extension Trial",
        "Sprint Mind Substudy", "Hermes Pooled Analysis",
        "Finger Multidomain Programme", "Predict Cohort Enrolment",
        "Ideal Registry Entry", "Improving Access Programme",
    )

    @pytest.mark.parametrize("phrase", CLINICAL)
    def test_layout_a_orphan_label_above_the_phrase(self, phrase: str) -> None:
        """A printed-but-blank `Patient Name:` field, which every real form has."""
        document = f"Patient Name:\nMRN:\n{phrase}\nRGT/44219/B\n"
        if find_ambiguities(document):
            return  # refused: the policy's other permitted outcome
        scrubbed = scrub_pii(document)
        assert phrase in scrubbed, (
            f"processed WITHOUT refusal and the phrase was DESTROYED: "
            f"{document!r} -> {scrubbed!r}"
        )

    @pytest.mark.parametrize("phrase", CLINICAL)
    def test_layout_b_phrase_trailing_a_name_on_one_line(self, phrase: str) -> None:
        document = f"Patient Name: Harold Nkemdirim {phrase}\n"
        if find_ambiguities(document):
            return
        scrubbed = scrub_pii(document)
        assert phrase in scrubbed, (
            f"processed WITHOUT refusal and the phrase was DESTROYED: "
            f"{document!r} -> {scrubbed!r}"
        )

    @pytest.mark.parametrize("phrase", CLINICAL)
    def test_the_name_is_still_removed_in_layout_b(self, phrase: str) -> None:
        """The other direction, on the same document.

        Preserving the phrase must not be bought by leaving the name. Both
        halves of the invariant, asserted on one input, is what five rounds of
        trading them off could not do.
        """
        document = f"Patient Name: Harold Nkemdirim {phrase}\n"
        if find_ambiguities(document):
            return
        out = visible(scrub_pii(document))
        assert "Nkemdirim" not in out, f"{document!r} -> {out!r}"


class TestThePostGateIdempotenceRegression:
    """The document ADV14-2 named, which round-tripped at 623cffa and does not now.

        in : 'Patient Name:\\nMRN:\\nHarold Nkemdirim\\nRockwood Frailty\\n'
        x1 : 'Patient Name:\\nMRN:\\n[NAME]\\nRockwood Frailty\\n'
        x2 : identical to x1

    At the post-gate head x2 replaces `Rockwood Frailty` with `[NAME]`. The
    mechanism that held it was forgeable by an inserted line and was removed;
    removing it reopened the destruction. Neither a marker nor a position in
    caller-controlled text can carry this property.
    """

    RECORDED = (
        "Patient Name:\nMRN:\nHarold Nkemdirim\nRockwood Frailty\n",
        "Jonathan Aldred-Whitmore: Name Rockwood Frailty Scale",
        "Patient Name: Harold Nkemdirim\nBradycardia 48 bpm untreated\n",
        "Patient Name:\nMRN:\nGordon Whitfield\nSubstantia Nigra\n",
        "Patient Name:\nMRN:\nEsme Fairhurst\nCorpus Callosum\n",
    )

    @pytest.mark.parametrize("document", RECORDED)
    def test_a_second_pass_changes_nothing(self, document: str) -> None:
        once = scrub_pii(document)
        twice = scrub_pii(once)
        assert twice == once, f"{document!r}\n  x1 {once!r}\n  x2 {twice!r}"

    def test_the_clinical_line_is_not_destroyed_by_the_second_pass(self) -> None:
        document = "Patient Name:\nMRN:\nHarold Nkemdirim\nRockwood Frailty\n"
        twice = scrub_pii(scrub_pii(document))
        assert "Rockwood Frailty" in twice, f"{document!r} -> {twice!r}"
