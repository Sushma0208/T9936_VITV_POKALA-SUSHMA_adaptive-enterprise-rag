from app.verification.evidence_checker import EvidenceChecker


def create_source(text):
    return {
        "document": {
            "text": text,
            "source": "WFH_Policy.txt"
        }
    }


def main():

    checker = EvidenceChecker()

    evidence = [
        create_source(
            "Employees may work from home for a maximum "
            "of 8 days per month with manager approval."
        )
    ]

    # -----------------------------------------
    # Test 1: Supported answer
    # -----------------------------------------

    supported_answer = (
        "Employees can work from home for a maximum "
        "of 8 days per month with manager approval."
    )

    result = checker.check(
        answer=supported_answer,
        sources=evidence
    )

    print("\nTEST 1: Supported Answer")
    print("Answer:", supported_answer)
    print("Result:", result)

    assert result["supported"] is True


    # -----------------------------------------
    # Test 2: Hallucinated numeric claim
    # -----------------------------------------

    hallucinated_answer = (
        "Employees can work from home for "
        "25 days per month."
    )

    result = checker.check(
        answer=hallucinated_answer,
        sources=evidence
    )

    print("\nTEST 2: Unsupported Numeric Claim")
    print("Answer:", hallucinated_answer)
    print("Result:", result)

    assert result["supported"] is False


    # -----------------------------------------
    # Test 3: Completely unsupported answer
    # -----------------------------------------

    unsupported_answer = (
        "Employees receive free company housing."
    )

    result = checker.check(
        answer=unsupported_answer,
        sources=evidence
    )

    print("\nTEST 3: Unsupported Answer")
    print("Answer:", unsupported_answer)
    print("Result:", result)

    assert result["supported"] is False


    # -----------------------------------------
    # Test 4: Insufficient evidence response
    # -----------------------------------------

    insufficient_answer = (
        "Insufficient evidence in the available documents."
    )

    result = checker.check(
        answer=insufficient_answer,
        sources=evidence
    )

    print("\nTEST 4: Insufficient Evidence")
    print("Answer:", insufficient_answer)
    print("Result:", result)

    assert result["supported"] is False


    print("\n===================================")
    print("ALL EVIDENCE CHECKS PASSED")
    print("===================================")


if __name__ == "__main__":
    main()