"""Pre-publication hygiene for the legacy PMC ingestion path.

Two separate problems, both blocking a public repository:

1. ``mao/data/download_pmc.py`` carried a maintainer's personal email address as
   a literal default. Publishing it would publish the address.
2. That same module is the D1-D3 ingestion path the P2-0 audit disproved. It
   produced a corpus in which every license field read ``"open-access"`` — a
   broken-XPath fallback, not a license — and every citation field was empty.
   Left runnable in a public repo, a future user could reasonably assume it
   works, and would silently rebuild an unpublishable corpus.

The canonical path is ``mao/corpus/``. These tests pin that the old one refuses
to run and says where to go instead, and that it is still *importable*, because
``mao/data/reingest_all.py`` imports from it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PUBLISHABLE = ("mao", "scripts", "app")

#: Consumer mail providers. An address at one of these in source is a personal
#: contact detail, never a fixture or a de-identification example.
_PERSONAL_EMAIL = re.compile(
    r"[A-Za-z0-9._%+-]+@(?:gmail|yahoo|outlook|hotmail|icloud|proton|protonmail|aol)\."
    r"[A-Za-z]{2,}",
    re.IGNORECASE,
)

#: `os.getenv("...EMAIL...", "someone@somewhere")` — the exact defect shape:
#: a contact address that is used whenever the environment does not supply one.
_EMAIL_DEFAULT = re.compile(
    r"getenv\(\s*['\"][^'\"]*EMAIL[^'\"]*['\"]\s*,\s*['\"][^'\"]*@[^'\"]*['\"]",
    re.IGNORECASE,
)


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for package in _PUBLISHABLE:
        files.extend((_ROOT / package).rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


class TestNoPersonalContactDetailsArePublishable:
    def test_no_personal_email_in_any_publishable_source(self) -> None:
        offenders = [
            f"{path.relative_to(_ROOT).as_posix()}: {match.group(0)}"
            for path in _python_sources()
            for match in _PERSONAL_EMAIL.finditer(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert offenders == []

    def test_no_email_is_hardcoded_as_an_env_var_default(self) -> None:
        """A default contact address is used silently whenever the env is unset."""
        offenders = [
            path.relative_to(_ROOT).as_posix()
            for path in _python_sources()
            if _EMAIL_DEFAULT.search(path.read_text(encoding="utf-8", errors="ignore"))
        ]
        assert offenders == []


class TestLegacyIngestionRefusesToRun:
    def test_the_module_still_imports(self) -> None:
        """``mao/data/reingest_all.py`` imports from it; breaking that is out of scope."""
        import mao.data.download_pmc as legacy

        assert legacy is not None

    def test_downloading_raises_and_names_the_canonical_path(self) -> None:
        from mao.data.download_pmc import download_pmc_papers

        with pytest.raises(RuntimeError, match="mao.corpus"):
            download_pmc_papers()

    def test_ingesting_raises_and_names_the_canonical_path(self) -> None:
        from mao.data.download_pmc import ingest_pmc_papers

        with pytest.raises(RuntimeError, match="mao.corpus"):
            ingest_pmc_papers()

    def test_the_refusal_explains_why_rather_than_only_that(self) -> None:
        from mao.data.download_pmc import download_pmc_papers

        with pytest.raises(RuntimeError, match="(?i)licen[cs]e|provenance|open-access"):
            download_pmc_papers()

    def test_the_manifest_path_is_still_exported(self) -> None:
        """``reingest_all`` imports this symbol; it must survive the retirement."""
        from mao.data.download_pmc import _MANIFEST_PATH

        assert _MANIFEST_PATH.name == "pmc_manifest.json"


class TestTheModuleDocumentsItsOwnRetirement:
    def test_the_docstring_points_at_the_canonical_path(self) -> None:
        import mao.data.download_pmc as legacy

        assert "mao/corpus" in (legacy.__doc__ or "")

    def test_the_docstring_records_the_defects(self) -> None:
        import mao.data.download_pmc as legacy

        doc = legacy.__doc__ or ""
        assert "D1" in doc and "D2" in doc and "D3" in doc
