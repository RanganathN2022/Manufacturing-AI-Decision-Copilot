from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CHROMA_DIR = BASE_DIR / "chroma_db"


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# CONNECT TO CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="manufacturing_knowledge"
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_knowledge(
    query,
    top_k=2
):
    """
    Retrieve the most relevant maintenance
    knowledge chunks for a user query.
    """

    query_embedding = embedding_model.encode(
        [query]
    )

    results = collection.query(
        query_embeddings=[
            query_embedding[0].tolist()
        ],
        n_results=top_k
    )

    retrieved_documents = []

    for i, document in enumerate(
        results["documents"][0]
    ):

        retrieved_documents.append({
            "document": document,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    )

    results = retrieve_knowledge(query)

    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 60)

        print(
            "Source:",
            result["source"]
        )

        print(
            "Distance:",
            result["distance"]
        )

        print(
            "Content:"
        )

        print(
            result["document"]
        )