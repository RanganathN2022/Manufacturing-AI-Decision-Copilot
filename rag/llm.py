import ollama

from rag.retrieve import retrieve_knowledge


MODEL_NAME = "tinyllama:latest"


def generate_answer(question):

    retrieved_documents = retrieve_knowledge(
        question,
        top_k=2
    )

    # Combine only the retrieved content
    context = "\n\n".join(
        item["document"]
        for item in retrieved_documents
    )

    prompt = f"""
Use the following maintenance information to answer
the question.

MAINTENANCE INFORMATION:
{context}

QUESTION:
{question}

ANSWER:

Recommended inspection:
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0.0,
            "num_predict": 120
        }
    )

    answer = response["message"]["content"]

    return {
        "question": question,
        "retrieved_documents": retrieved_documents,
        "answer": answer
    }


if __name__ == "__main__":

    question = (
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    )

    result = generate_answer(question)

    print("=" * 60)
    print("RAG + LLM TEST")
    print("=" * 60)

    print("\nQUESTION:")
    print(result["question"])

    print("\nRETRIEVED SOURCES:")

    for document in result["retrieved_documents"]:
        print(
            f"- {document['source']}"
        )

    print("\nLLM ANSWER:")
    print(result["answer"])