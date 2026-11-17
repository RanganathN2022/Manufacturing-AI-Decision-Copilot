
from pathlib import Path
import re
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "manufacturing_knowledge"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD EMBEDDING MODEL (REUSE)
# ============================================================

@lru_cache(maxsize=1)
def get_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    print("Embedding model loaded.")

    return model


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: {KNOWLEDGE_DIR}"
        )

    for file_path in sorted(
        KNOWLEDGE_DIR.glob("*.txt")
    ):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:

            documents.append({
                "source": file_path.name,
                "text": text
            })

    if not documents:
        raise ValueError(
            "No valid knowledge documents found."
        )

    print(
        f"Documents loaded: {len(documents)}"
    )

    return documents


# ============================================================
# SECTION-BASED CHUNKING
# ============================================================

def create_chunks(documents):

    chunks = []

    for document in documents:

        text = document["text"]

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

    if not chunks:
        raise ValueError(
            "No document chunks were generated."
        )

    print(
        f"Chunks created: {len(chunks)}"
    )

    return chunks


# ============================================================
# CHROMADB CLIENT
# ============================================================

@lru_cache(maxsize=1)
def get_chroma_client():

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )


# ============================================================
# CREATE OR RETRIEVE COLLECTION
# ============================================================

def get_or_create_knowledge_collection():

    client = get_chroma_client()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Manufacturing maintenance knowledge"
        }
    )

    # Check whether the collection already contains data.

    if collection.count() > 0:

        print(
            f"Existing ChromaDB collection found: "
            f"{collection.count()} records"
        )

        return collection

    # Collection is missing or empty.
    # Automatically ingest knowledge documents.

    print(
        "ChromaDB collection is empty."
    )

    print(
        "Starting automatic knowledge ingestion..."
    )

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    embedding_model = get_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False
    )

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
        collection.count()
    )

    print(
        "Collection:",
        collection.name
    )

    return collection


# ============================================================
# MANUAL INGESTION TEST
# ============================================================

if __name__ == "__main__":

    collection = get_or_create_knowledge_collection()

    print("\nINGESTION STATUS")
    print("------------------------------")

    print(
        "Collection:",
        collection.name
    )

    print(
        "Total records:",
        collection.count()
    )
