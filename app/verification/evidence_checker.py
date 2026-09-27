import re
from typing import List, Dict


class EvidenceChecker:

    def check(
        self,
        answer: str,
        sources: List[Dict]
    ) -> Dict:

        if not sources:

            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "No supporting evidence was retrieved."
            }

        evidence_text = " ".join(
            source["document"]["text"]
            for source in sources
        )

        # ---------------------------------------
        # 1. Check numeric consistency
        # ---------------------------------------

        answer_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            answer
        )

        evidence_numbers = re.findall(
            r"\b\d+(?:\.\d+)?\b",
            evidence_text
        )

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

        # ---------------------------------------
        # 2. Calculate lexical evidence overlap
        # ---------------------------------------

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

        if not answer_words:

            return {
                "supported": False,
                "confidence": 0.0,
                "reason": "Generated answer is empty."
            }

        overlap = answer_words.intersection(
            evidence_words
        )

        confidence = len(overlap) / len(answer_words)

        supported = confidence >= 0.5

        return {
            "supported": supported,
            "confidence": round(confidence, 3),
            "reason": (
                "Answer contains sufficient evidence overlap."
                if supported
                else "Answer contains insufficient evidence overlap."
            )
        }