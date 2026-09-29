from pathlib import Path
import json
from pypdf import PdfReader

DOCUMENTS_DIR = Path("documents")
METADATA_FILE = DOCUMENTS_DIR / "metadata.json"


def load_metadata():
    if not METADATA_FILE.exists():
        return {}

    return json.loads(
        METADATA_FILE.read_text(encoding="utf-8")
    )


def extract_access_roles(text):
    for line in text.splitlines():
        if line.startswith("Access Level:"):
            roles = line.split(":", 1)[1].strip()
            return [role.strip() for role in roles.split(",")]

    return []


def load_text_file(file_path, metadata):
    text = file_path.read_text(encoding="utf-8")

    relative_path = file_path.relative_to(DOCUMENTS_DIR).as_posix()
    file_metadata = metadata.get(relative_path, {})

    return [{
        "text": text,
        "source": file_path.name,
        "file_type": "txt",
        "organization": file_metadata.get(
            "organization",
            file_path.parent.name
        ),
        "document_type": file_metadata.get(
            "document_type",
            "Internal Document"
        ),
        "page_number": None,
        "allowed_roles": extract_access_roles(text),
        "access_type": file_metadata.get(
            "access_type",
            "restricted"
        ),
    }]


def load_pdf_file(file_path, metadata):
    reader = PdfReader(str(file_path))

    relative_path = file_path.relative_to(DOCUMENTS_DIR).as_posix()
    file_metadata = metadata.get(relative_path, {})

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if not text.strip():
            continue

        documents.append({
            "text": text,
            "source": file_path.name,
            "file_type": "pdf",
            "organization": file_metadata.get(
                "organization",
                file_path.parent.name
            ),
            "document_type": file_metadata.get(
                "document_type",
                "Public Policy"
            ),
            "page_number": page_number,
            "allowed_roles": [],
            "access_type": file_metadata.get(
                "access_type",
                "public"
            ),
        })

    return documents


def load_documents():
    documents = []
    metadata = load_metadata()

    for file_path in DOCUMENTS_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.name == "metadata.json":
            continue

        if file_path.suffix.lower() == ".txt":
            documents.extend(
                load_text_file(file_path, metadata)
            )

        elif file_path.suffix.lower() == ".pdf":
            documents.extend(
                load_pdf_file(file_path, metadata)
            )

    return documents


def load_text_documents():
    return load_documents()