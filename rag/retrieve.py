
from rag.ingest import (
    get_embedding_model,
    get_or_create_knowledge_collection
)


# ============================================================
# INITIALIZE RAG
# ============================================================

print("Initializing Manufacturing Knowledge Assistant...")

# Automatically creates and populates collection if needed.

collection = get_or_create_knowledge_collection()

# Reuse the same embedding model.

embedding_model = get_embedding_model()

print("RAG initialization successful.")


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_knowledge(
    query,
    top_k=2
):
    """
    Retrieve relevant manufacturing maintenance
    knowledge using semantic similarity.
    """

    if not query or not query.strip():

        raise ValueError(
            "Search query cannot be empty."
        )

    if top_k < 1:

        raise ValueError(
            "top_k must be at least 1."
        )

    total_records = collection.count()

    if total_records == 0:

        raise RuntimeError(
            "ChromaDB collection contains no documents."
        )

    # Generate query embedding.

    query_embedding = embedding_model.encode(
        [query]
    )

    # Retrieve relevant documents.

    results = collection.query(

        query_embeddings=[
            query_embedding[0].tolist()
        ],

        n_results=min(
            top_k,
            total_records
        ),

        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_documents = []

    documents = results.get(
        "documents", [[]]
    )[0]

    metadatas = results.get(
        "metadatas", [[]]
    )[0]

    distances = results.get(
        "distances", [[]]
    )[0]

    for index, document in enumerate(documents):

        metadata = (
            metadatas[index] or {}
        )

        retrieved_documents.append({

            "document": document,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "distance": distances[index]

        })

    return retrieved_documents


# ============================================================
# RETRIEVAL TEST
# ============================================================

if __name__ == "__main__":

    query = (
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    )

    results = retrieve_knowledge(
        query,
        top_k=2
    )

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nResult {index}"
        )

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
