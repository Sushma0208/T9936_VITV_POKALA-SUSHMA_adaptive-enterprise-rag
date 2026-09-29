import re
from typing import List, Dict


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient evidence in the available documents."
)


class EvidenceChecker:

    def check(
        self,
        answer: str,
        sources: List[Dict]
    ) -> Dict:

        # ---------------------------------------------------------
        # 1. No evidence retrieved
        # ---------------------------------------------------------
        if not sources:
            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "No supporting evidence was retrieved."
            }

        # ---------------------------------------------------------
        # 2. LLM already returned the fallback response
        # ---------------------------------------------------------
        if answer.strip() == INSUFFICIENT_EVIDENCE_MESSAGE:
            return {
                "supported": False,
                "confidence": 0.0,
                "reason": (
                    "The model determined that the available "
                    "evidence was insufficient."
                )
            }

        # ---------------------------------------------------------
        # 3. Combine retrieved evidence
        # ---------------------------------------------------------
        evidence_text = " ".join(
            source["document"]["text"]
            for source in sources
        )

        # ---------------------------------------------------------
        # 4. Remove page/citation references before
        #    checking numerical factual claims
        # ---------------------------------------------------------
        answer_for_numeric_check = re.sub(
            r"\bpages?\s+\d+(?:\s*[-–]\s*\d+)?",
            "",
            answer,
            flags=re.IGNORECASE
        )

        # ---------------------------------------------------------
        # 5. Extract numerical claims
        # ---------------------------------------------------------
        answer_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            answer_for_numeric_check
        )

        evidence_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            evidence_text
        )

        # ---------------------------------------------------------
        # 6. Verify numerical claims
        # ---------------------------------------------------------
        for number in answer_numbers:

            if number not in evidence_numbers:

                return {
                    "supported": False,
                    "confidence": 0.0,
                    "reason": (
                        f"Answer contains numeric claim "
                        f"'{number}' not found in the evidence."
                    )
                }

        # ---------------------------------------------------------
        # 7. Extract words from answer
        # ---------------------------------------------------------
        answer_words = set(
            re.findall(
                r"\b[a-zA-Z]+\b",
                answer.lower()
            )
        )

        evidence_words = set(
            re.findall(
                r"\b[a-zA-Z]+\b",
                evidence_text.lower()
            )
        )

        # ---------------------------------------------------------
        # 8. Empty answer check
        # ---------------------------------------------------------
        if not answer_words:

            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "Generated answer is empty."
            }

        # ---------------------------------------------------------
        # 9. Calculate evidence overlap
        # ---------------------------------------------------------
        overlap = answer_words.intersection(
            evidence_words
        )

        confidence = (
            len(overlap) / len(answer_words)
        )

        # ---------------------------------------------------------
        # 10. Determine support
        # ---------------------------------------------------------
        supported = confidence >= 0.5

        return {
            "supported": supported,
            "confidence": round(
                confidence,
                3
            ),
            "reason": (
                "Answer contains sufficient evidence overlap."
                if supported
                else
                "Answer contains insufficient evidence overlap."
            )
        }