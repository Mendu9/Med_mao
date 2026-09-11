r"""ADV16-1's second half: the wall shared the scrubber's blind spot.

The finding was not only that `scrub_pii` missed a decorated postcode. It was
that `RequestProtection.leaked_in` missed it too, because `gateway._visible`
applied NFKC without NFD exactly as `normalise` did. The run-scoped identifier
assertion is the control the gateway module itself calls "the one that closes
the class of defect the previous six waves kept reopening", and both walls
shared one detection step, so neither could catch what the other missed.

`gateway._visible` and `mao.core.deident.text.normalise` are therefore
deliberately DIFFERENT and must not be refactored into one shared helper:

    normalise()        a TRANSFORMATION whose output is the text that ships.
                       It must preserve marks that carry meaning.
    gateway._visible() a COMPARISON that never ships anything. Over-normalising
                       costs nothing and only widens what the wall catches, so
                       it strips every mark after decomposing.

This file deliberately does not import `mao.core.deident.text`.
"""
from __future__ import annotations

from mao.trust.classes import InputChannel
from mao.trust.egress.gateway import RequestProtection, _visible


class TestTheBackstopSeesThroughAnyMark:
    def test_a_decorated_postcode_is_still_read_as_the_postcode(self) -> None:
        assert "SW1A 1AA" in _visible("Postcode: SW1́A 1AA")

    def test_a_decorated_record_number_is_still_read(self) -> None:
        assert "A1234567" in _visible("MRN: Á1234567")

    def test_a_decorated_ni_number_is_still_read(self) -> None:
        assert "QQ123456C" in _visible("NI: QQ́123456C")

    def test_it_strips_marks_the_scrubber_deliberately_preserves(self) -> None:
        """The intentional asymmetry, asserted so a later refactor cannot
        quietly make the two functions one again."""
        assert _visible("प्र") == "पर"

    def test_the_zero_width_property_it_already_had_still_holds(self) -> None:
        assert "Nkemdirim" in _visible("Nkem͏dirim")


class TestTheAssertionFiresOnADecoratedIdentifier:
    def test_leaked_in_reports_a_decorated_form_of_a_removed_identifier(self) -> None:
        protection = RequestProtection(trace_id="t")
        protection.record_identifier("POSTCODE", "SW1A 1AA", InputChannel.QUERY)
        assert protection.leaked_in("home postcode SW1́A 1AA") == [
            "POSTCODE:query"
        ]

    def test_it_reports_a_decorated_record_number_too(self) -> None:
        protection = RequestProtection(trace_id="t")
        protection.record_identifier("MRN", "A1234567", InputChannel.REPORT)
        assert protection.leaked_in("record Á1234567") == ["MRN:report"]

    def test_it_does_not_fire_when_nothing_was_removed(self) -> None:
        """Non-vacuity. A wall that reports a leak on an empty record is not
        measuring the text."""
        protection = RequestProtection(trace_id="t")
        assert protection.leaked_in("home postcode SW1́A 1AA") == []

    def test_it_does_not_fire_on_unrelated_text(self) -> None:
        protection = RequestProtection(trace_id="t")
        protection.record_identifier("POSTCODE", "SW1A 1AA", InputChannel.QUERY)
        assert protection.leaked_in("donepezil 10 mg once daily") == []
