def build_rag_prompt(question, retrieved_documents):
    evidence = []

    for result in retrieved_documents:
        document = result["document"]

        evidence.append(
            f"""
Source: {document["source"]}
Department: {document["department"]}

Content:
{document["text"]}
"""
        )

    context = "\n".join(evidence)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY the retrieved enterprise
documents provided below.

If the retrieved documents do not contain enough information
to answer the question, respond exactly with:

"Insufficient evidence in the available documents."

Do not invent information.
Do not use outside knowledge.
Keep the answer concise and factual.

Retrieved Enterprise Evidence:
{context}

User Question:
{question}

Answer:
"""

    return prompt