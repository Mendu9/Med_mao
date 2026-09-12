r"""Control decision M-3 — the MRI prediction workflow is RETIRED.

These tests prove REACHABILITY, not UI absence. A tab can be deleted while the
route behind it still answers, and an agent function can be renamed while the
model it loads is still downloaded at boot; neither of those is a retirement.

So each assertion below binds to the thing that would actually have to be true
for the workflow to still run: the module is gone from the import system, no
module under `mao/` names it, no dispatch reaches a handler, and — the ADV17-5
regression — no request carrying a caller-supplied image URL performs an
outbound fetch.

## Why the fetch assertion is the important one

`ff34722`'s adversarial review recorded ADV17-5: `_run_mri_prediction` called
`fetch_image_bytes(metadata["image_url"])` BEFORE any image gate was consulted.
M-2 had already shut image EGRESS, so the system refused to send a scan to a
model while still fetching a caller-named URL with the deployment's own network
position. The gate was on the wrong side of the I/O.

Deleting the caller is what closes that structurally. `mao/safety/fetch.py`
survives, because `multimodal_agent.handle_image` still calls it — but that
call sits AFTER the `IMAGE_ENABLED` refusal, which returns before any I/O.

## On non-vacuity

Every instrument here is proven to fire before it is used to prove an absence.
A network recorder that was never wired up correctly records zero calls on a
build where the feature still works, and would have passed at `ff34722`. Each
such control is marked `NON-VACUITY CONTROL` below.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MAO = REPO_ROOT / "mao"

_RETIRED_MODULE = "mao.models.mri_predictor"

#: base64 "x" — a payload shaped like an attachment, carrying nothing.
_IMAGE_B64 = "eA=="
_IMAGE_URL = "https://example.invalid/scan.jpg"


# ---------------------------------------------------------------------------
# 1. The module is gone from the import system, not merely unused
# ---------------------------------------------------------------------------

class TestThePredictorModuleIsGone:
    def test_it_is_not_importable(self) -> None:
        with pytest.raises(ImportError):
            importlib.import_module(_RETIRED_MODULE)

    def test_its_file_is_not_on_disk(self) -> None:
        """Distinct from the import test: a stale .pyc or a namespace-package
        shadow could in principle satisfy one and not the other."""
        assert not (MAO / "models" / "mri_predictor.py").exists()

    def test_the_import_test_is_not_passing_on_a_typo(self) -> None:
        """NON-VACUITY CONTROL. `pytest.raises(ImportError)` passes for ANY
        unimportable name, including a misspelling of a module that still
        exists. A sibling module under the same package must still import, so
        the failure above is the module's absence rather than a broken path.
        """
        assert importlib.import_module("mao.agents.clinical_agent") is not None
        with pytest.raises(ImportError):
            importlib.import_module("mao.models.this_module_never_existed")


# ---------------------------------------------------------------------------
# 2. Nothing under mao/ imports it
# ---------------------------------------------------------------------------

def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module name imported by *path*, including inside functions."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:  # pragma: no cover - the repo is expected to parse
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class TestNoModuleImportsThePredictor:
    def test_no_source_file_under_mao_imports_it(self) -> None:
        """Parsed, not grepped. A grep would be satisfied by the comment in
        `clinical_agent` that records this retirement."""
        offenders = [
            path.relative_to(REPO_ROOT).as_posix()
            for path in MAO.rglob("*.py")
            if any(
                name == _RETIRED_MODULE or name.startswith(_RETIRED_MODULE + ".")
                for name in _imported_modules(path)
            )
        ]
        assert offenders == [], f"still importing the retired predictor: {offenders}"

    def test_the_import_scan_would_notice_a_real_import(self) -> None:
        """NON-VACUITY CONTROL. The scanner must detect both import forms,
        including one nested inside a function — which is exactly how
        `_run_mri_prediction` and the API warm-up imported it. A scanner that
        only walked module-level imports would have reported zero offenders on
        the build where the defect existed.
        """
        probe = ast.parse(
            "def f():\n"
            "    from mao.models.mri_predictor import get_predictor\n"
            "    import mao.models.mri_predictor\n"
        )
        names: set[str] = set()
        for node in ast.walk(probe):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        assert _RETIRED_MODULE in names

    def test_importing_the_api_pulls_in_no_tensorflow_or_keras(self) -> None:
        """The warm-up in `mao/api/main.py` imported the predictor at startup,
        which imported tensorflow and could download a 127 MB checkpoint. The
        binding is `sys.modules` after importing the API, because that is what
        a boot actually does.
        """
        importlib.import_module("mao.api.main")
        heavy = sorted(
            name
            for name in sys.modules
            if name == "keras"
            or name == "tensorflow"
            or name.startswith(("tensorflow.", "keras."))
        )
        assert heavy == [], f"importing the API pulled in {heavy}"


# ---------------------------------------------------------------------------
# 3. No dispatch reaches an MRI handler
# ---------------------------------------------------------------------------

class TestNoDispatchReachesAnMriHandler:
    @pytest.mark.parametrize(
        "attribute",
        [
            "_handle_mri_image",
            "_run_mri_prediction",
            "_format_prediction",
            "_interpret_stage",
            "_describe_image",
        ],
    )
    def test_the_handler_is_not_an_attribute_of_the_agent(
        self, attribute: str
    ) -> None:
        from mao.agents import clinical_agent

        assert not hasattr(clinical_agent, attribute)

    def test_the_attribute_check_is_bound_to_the_right_module(self) -> None:
        """NON-VACUITY CONTROL. `hasattr` is False for a module that failed to
        import, for a typo'd module name, and for a module that never had the
        attribute. Asserting a function that IS still there proves the handle
        points at the live clinical agent."""
        from mao.agents import clinical_agent

        assert hasattr(clinical_agent, "clinical_node")
        assert hasattr(clinical_agent, "_handle_pdf_report")

    def test_the_policy_no_longer_carries_the_confidence_gate(self) -> None:
        from mao.safety.policy import get_policy

        assert not hasattr(get_policy(), "mri_confidence_gate")


# ---------------------------------------------------------------------------
# 4. ADV17-5 — an image request performs no outbound fetch
# ---------------------------------------------------------------------------

@pytest.fixture()
def network_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Instrument every real network seam a fetch could leave through.

    Patched at MODULE level rather than at the agent, so a call made through
    any import alias is still caught: `requests.get`, `httpx.get`/`request`,
    `urllib.request.urlopen`, and `mao.safety.fetch.fetch_image_bytes` itself.
    """
    seen: list[tuple[str, str]] = []

    def _record(label: str):
        def _hook(*args: object, **kwargs: object):
            seen.append((label, str(args[0]) if args else ""))
            raise AssertionError(
                f"an outbound fetch was attempted via {label}: {args!r}"
            )

        return _hook

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _record("urllib.urlopen"))

    import requests

    monkeypatch.setattr(requests, "get", _record("requests.get"))
    monkeypatch.setattr(requests, "request", _record("requests.request"))

    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx is installed here
        pass
    else:
        monkeypatch.setattr(httpx, "get", _record("httpx.get"))
        monkeypatch.setattr(httpx, "request", _record("httpx.request"))

    import mao.safety.fetch as fetch_module

    monkeypatch.setattr(
        fetch_module, "fetch_image_bytes", _record("safety.fetch.fetch_image_bytes")
    )
    return seen


class TestAnImageRequestPerformsNoOutboundFetch:
    """ADV17-5 regression."""

    @pytest.mark.parametrize(
        "metadata",
        [
            {"image_url": _IMAGE_URL},
            {"image_b64": _IMAGE_B64},
            {"image_url": _IMAGE_URL, "image_b64": _IMAGE_B64},
            # The metadata-server target the guard exists to refuse. It must
            # not even be attempted, which is a stronger claim than refusing it.
            {"image_url": "http://169.254.169.254/latest/meta-data/"},
        ],
    )
    def test_clinical_node_makes_no_fetch(
        self, metadata: dict, network_calls: list
    ) -> None:
        from mao.agents.clinical_agent import clinical_node

        state = clinical_node(
            {
                "user_query": "Does this scan show atrophy?",
                "user_id": "adv17-5",
                "metadata": dict(metadata),
                "memory_context": "",
            }
        )

        assert network_calls == [], f"outbound fetch attempted: {network_calls}"
        # The request still had to be ANSWERED. A crash would also produce zero
        # fetches, and would not be a retirement.
        assert state["response"].strip()

    def test_the_instrument_fires_when_a_fetch_really_happens(
        self, network_calls: list
    ) -> None:
        """NON-VACUITY CONTROL — the most important test in this file.

        Every assertion above is an absence recorded by this fixture. If the
        fixture patched the wrong names, it would record zero calls on a build
        where `_run_mri_prediction` still fetched, and the whole class would
        pass while ADV17-5 was wide open.

        So: call each patched seam deliberately and require that it is seen.
        """
        import urllib.request

        import requests

        import mao.safety.fetch as fetch_module

        for call in (
            lambda: requests.get(_IMAGE_URL),
            lambda: urllib.request.urlopen(_IMAGE_URL),
            lambda: fetch_module.fetch_image_bytes(_IMAGE_URL),
        ):
            with pytest.raises(AssertionError):
                call()

        labels = {label for label, _ in network_calls}
        assert labels == {
            "requests.get",
            "urllib.urlopen",
            "safety.fetch.fetch_image_bytes",
        }, f"the instrument missed a seam; saw {labels}"

    def test_the_surviving_fetch_helper_has_exactly_one_caller(self) -> None:
        """`fetch_image_bytes` is NOT removed: `multimodal_agent.handle_image`
        still calls it. What closes ADV17-5 is that its last remaining caller
        sits behind the M-2 refusal, whereas `_run_mri_prediction` sat in front
        of it. Asserted by parsing, so the comment recording ADV17-5 in
        `clinical_agent` does not satisfy it.
        """
        callers = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in MAO.rglob("*.py")
            if path.name != "fetch.py"
            and any(
                isinstance(node, ast.Call)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id == "fetch_image_bytes")
                    or (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr == "fetch_image_bytes"
                    )
                )
                for node in ast.walk(
                    ast.parse(path.read_text(encoding="utf-8", errors="replace"))
                )
            )
        )
        assert callers == ["mao/agents/multimodal_agent.py"], (
            f"unexpected fetch_image_bytes callers: {callers}"
        )


# ---------------------------------------------------------------------------
# 5. The image attachment path returns the M-2 refusal
# ---------------------------------------------------------------------------

class TestTheImagePathReturnsTheM2Refusal:
    def test_an_image_attachment_is_told_the_modality_is_unavailable(self) -> None:
        from mao.agents.clinical_agent import clinical_node

        state = clinical_node(
            {
                "user_query": "Stage this scan.",
                "user_id": "m2",
                "metadata": {"image_b64": _IMAGE_B64},
                "memory_context": "",
            }
        )
        assert "image analysis is not available" in state["response"].lower()

    def test_the_refusal_is_the_one_multimodal_agent_owns(self) -> None:
        """One implementation, not a second copy pasted into the clinical
        agent - which is the drift `multimodal_agent` exists to prevent."""
        from mao.agents.clinical_agent import clinical_node
        from mao.agents.multimodal_agent import handle_image

        owned, meta = handle_image("Stage this scan.", {"image_b64": _IMAGE_B64}, "")
        assert meta.get("error") == "image_disabled_phase_1"

        state = clinical_node(
            {
                "user_query": "Stage this scan.",
                "user_id": "m2",
                "metadata": {"image_b64": _IMAGE_B64},
                "memory_context": "",
            }
        )
        assert owned.strip() in state["response"]

    def test_the_image_branch_claims_the_request(self) -> None:
        """NON-VACUITY CONTROL for the two tests above. If the `has_image`
        branch had been DELETED rather than repointed, an image would fall
        through to the text path - which also returns a non-empty answer, and
        would satisfy a naive "it responded" assertion while silently dropping
        the attachment. The mode must not be the text path's.
        """
        from mao.agents.clinical_agent import clinical_node

        state = clinical_node(
            {
                "user_query": "Stage this scan.",
                "user_id": "m2",
                "metadata": {"image_b64": _IMAGE_B64},
                "memory_context": "",
            }
        )
        assert state["metadata"].get("mode") == "image"

    def test_neither_m2_wall_was_weakened(self) -> None:
        """M-3 must not have bought its refusal by relaxing M-2. Both walls are
        re-asserted here: the flag, and the egress policy row's absence."""
        from mao.agents import multimodal_agent
        from mao.trust.classes import TrustClass
        from mao.trust.egress.gateway import EgressRefused, authorise
        from mao.trust.egress.policy import Destination, EgressPurpose

        assert multimodal_agent.IMAGE_ENABLED is False
        with pytest.raises(EgressRefused, match="no approved flow"):
            authorise(
                destination=Destination.MODEL_PROVIDER,
                purpose=EgressPurpose.IMAGE_ANALYSIS,
                trust_class=TrustClass.SAFE_DERIVED_TEXT,
            )


# ---------------------------------------------------------------------------
# 6. The report card no longer reports a stage it did not measure
# ---------------------------------------------------------------------------

class TestTheCardDoesNotClaimAStage:
    def test_the_card_survives_but_reports_no_stage_and_no_confidence(self) -> None:
        """The ReportCard is kept - other modes use it. What must not survive
        is the retired path's fake default: `confidence_score=1.0` whenever
        there was no prediction, which rendered as "100%" on an answer that
        measured nothing.
        """
        from mao.agents.clinical_agent import clinical_node

        state = clinical_node(
            {
                "user_query": "Stage this scan.",
                "user_id": "card",
                "metadata": {"image_b64": _IMAGE_B64},
                "memory_context": "",
            }
        )
        card = state["report_card"]
        assert card["confidence_score"] is None, (
            "the card asserted a confidence nothing measured"
        )
        assert card["stage"] == "Not assessed"
        assert card["uncertainty_flag"] is False
