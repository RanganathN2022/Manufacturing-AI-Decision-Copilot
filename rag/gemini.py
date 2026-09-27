import os

from dotenv import load_dotenv
from google import genai

from rag.retrieve import retrieve_knowledge


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. "
        "Check your .env file."
    )

client = genai.Client(
    api_key=API_KEY
)

MODEL_NAME = "gemini-3.5-flash-lite"


# ============================================================
# GEMINI
# ============================================================

def generate_answer(prompt):
    """
    Send a prepared prompt to Gemini.
    """

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text

    prompt = f"""
You are a Manufacturing Maintenance AI Assistant.

Answer the user's question using the retrieved
maintenance knowledge provided below.

IMPORTANT RULES:

1. Use the retrieved knowledge as the primary source.
2. Do not invent maintenance procedures.
3. Do not claim that a machine will definitely fail.
4. If the knowledge is insufficient, say so.
5. Do not repeat the entire source documents.
6. Give a concise practical answer.

USER QUESTION:
{question}

RETRIEVED MAINTENANCE KNOWLEDGE:
{knowledge_context}

RESPONSE FORMAT:

Recommended inspection:

- Inspection item
- Inspection item
- Inspection item
- Inspection item

Reason:

Briefly explain why these inspections are relevant.

Keep the answer below 150 words.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# RAG + GEMINI TEST
# ============================================================

def answer_question(question):
    """
    Retrieve relevant knowledge and generate
    a grounded answer.
    """

    retrieved_documents = retrieve_knowledge(
        question,
        top_k=2
    )

    answer = generate_answer(
        question,
        retrieved_documents
    )

    return {
        "question": question,
        "retrieved_documents": retrieved_documents,
        "answer": answer
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = (
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    )

    result = answer_question(
        question
    )

    print("=" * 60)
    print("RAG + GEMINI TEST")
    print("=" * 60)

    print("\nQUESTION:")
    print(result["question"])

    print("\nRETRIEVED SOURCES:")

    for document in result[
        "retrieved_documents"
    ]:
        print(
            f"- {document['source']}"
        )

    print("\nGEMINI ANSWER:")
    print(result["answer"])