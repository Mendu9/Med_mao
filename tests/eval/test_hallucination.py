import pytest

KNOWN_FALSE_CLAIMS = [
    (
        "AD is caused by amyloid plaques and tau tangles.",
        "Alzheimer's disease is caused by a bacterial infection.",
    ),
    (
        "Donepezil is a symptomatic treatment with no cure for AD.",
        "Donepezil permanently cures Alzheimer's disease.",
    ),
    (
        "tPA must be given within 4.5 hours of stroke onset.",
        "tPA is effective when given 48 hours after stroke onset.",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("premise,claim", KNOWN_FALSE_CLAIMS)
def test_false_claim_not_entailed(premise: str, claim: str) -> None:
    """CrossEncoder NLI correctly identifies known-false medical claims."""
    from mao.eval.nli_checker import check_claim
    result = check_claim(premise=premise, claim=claim)
    assert result["entailed"] is False, (
        f"Expected FALSE claim to be unentailed.\n"
        f"Premise: {premise}\nClaim: {claim}\nScore: {result['score']}"
    )
