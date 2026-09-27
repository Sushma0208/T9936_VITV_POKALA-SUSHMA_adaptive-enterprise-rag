from pathlib import Path


DOCUMENTS_DIR = Path("documents")


def load_text_documents():
    """
    Load all TXT documents from the documents directory.

    Returns:
        list[dict]: Each document contains:
            - text
            - source
            - department
            - allowed_roles
    """

    documents = []

    for file_path in DOCUMENTS_DIR.rglob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        # Department is taken from the folder name
        department = file_path.parent.name

        # Extract access level from the document itself
        allowed_roles = []

        for line in text.splitlines():
            if line.startswith("Access Level:"):
                roles = line.split(":", 1)[1].strip()
                allowed_roles = [
                    role.strip()
                    for role in roles.split(",")
                ]
                break

        documents.append(
            {
                "text": text,
                "source": file_path.name,
                "department": department,
                "allowed_roles": allowed_roles,
            }
        )

    return documents