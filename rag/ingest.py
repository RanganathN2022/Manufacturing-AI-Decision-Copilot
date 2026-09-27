from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"

CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# LOAD DOCUMENTS
# ============================================================

documents = []

for file_path in KNOWLEDGE_DIR.glob("*.txt"):

    text = file_path.read_text(
        encoding="utf-8"
    )

    documents.append({
        "source": file_path.name,
        "text": text
    })


print(
    f"Documents loaded: {len(documents)}"
)


# ============================================================
# SECTION-BASED CHUNKING
# ============================================================

chunks = []

for document in documents:

    text = document["text"]

    # Split documents at major section headings.
    sections = re.split(
        r"\n(?=[A-Z][A-Z\s]+(?:\n|$))",
        text
    )

    section_index = 0

    for section in sections:

        section = section.strip()

        if not section:
            continue

        chunks.append({
            "id": (
                f"{document['source']}"
                f"_{section_index}"
            ),
            "source": document["source"],
            "text": section
        })

        section_index += 1

print(
    f"Chunks created: {len(chunks)}"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded"
)


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

texts = [
    chunk["text"]
    for chunk in chunks
]

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)


print(
    "Embeddings created:",
    embeddings.shape
)


# ============================================================
# CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="manufacturing_knowledge"
)


# ============================================================
# STORE DOCUMENTS
# ============================================================

collection.upsert(
    ids=[
        chunk["id"]
        for chunk in chunks
    ],

    documents=texts,

    embeddings=[
        embedding.tolist()
        for embedding in embeddings
    ],

    metadatas=[
        {
            "source": chunk["source"]
        }
        for chunk in chunks
    ]
)


print("=" * 50)
print("RAG INGESTION COMPLETE")
print("=" * 50)

print(
    "Documents:",
    len(documents)
)

print(
    "Chunks:",
    len(chunks)
)

print(
    "Stored in:",
    CHROMA_DIR
)

print(
    "Collection:",
    collection.name
)