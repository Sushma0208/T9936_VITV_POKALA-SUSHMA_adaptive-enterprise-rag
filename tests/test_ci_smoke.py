from app.retrieval.role_filter import filter_by_role
from app.verification.evidence_checker import EvidenceChecker


def make_result(text, access_type="public", allowed_roles=None):
    document = {
        "text": text,
        "source": "test_policy.txt",
        "access_type": access_type,
        "allowed_roles": allowed_roles or [],
    }
    return {"document": document, "score": 1.0}


def test_public_document_is_available_to_employee():
    results = [
        make_result("Public company policy")
    ]

    filtered = filter_by_role(results, "Employee")

    assert len(filtered) == 1


def test_restricted_document_is_hidden_from_unauthorized_role():
    results = [
        make_result(
            "Confidential HR policy",
            access_type="restricted",
            allowed_roles=["HR"],
        )
    ]

    filtered = filter_by_role(results, "Employee")

    assert filtered == []


def test_authorized_role_can_access_restricted_document():
    results = [
        make_result(
            "Confidential HR policy",
            access_type="restricted",
            allowed_roles=["HR"],
        )
    ]

    filtered = filter_by_role(results, "HR")

    assert len(filtered) == 1


def test_evidence_checker_rejects_unsupported_numeric_claim():
    sources = [
        make_result(
            "Employees can work from home for 8 days per month."
        )
    ]

    result = EvidenceChecker().check(
        answer="Employees can work from home for 25 days per month.",
        sources=sources,
    )

    assert result["supported"] is False
