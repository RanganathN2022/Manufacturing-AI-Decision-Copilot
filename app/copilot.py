from app.predict import predict_failure
from rag.retrieve import retrieve_knowledge
from rag.gemini import generate_answer


def run_copilot(
    machine_data,
    question,
    top_k=2
):
    """
    Run the complete Manufacturing AI Decision Copilot.

    Pipeline:

    Machine Data
        ↓
    ML Prediction
        ↓
    SHAP Explanation
        ↓
    RAG Retrieval
        ↓
    Gemini
        ↓
    Final AI Answer
    """

    # ========================================================
    # 1. ML PREDICTION
    # ========================================================

    prediction = predict_failure(
        machine_data
    )

    # ========================================================
    # 2. RAG RETRIEVAL
    # ========================================================

    retrieved_documents = retrieve_knowledge(
        question,
        top_k=top_k
    )

    # ========================================================
    # 3. BUILD AI CONTEXT
    # ========================================================

    probability = (
        prediction["failure_probability"]
    )

    risk_level = prediction[
        "risk_level"
    ]

    shap_explanation = prediction[
        "shap_explanation"
    ]

    # ========================================================
    # 4. BUILD GEMINI PROMPT
    # ========================================================

    knowledge_context = "\n\n".join(
        [
            (
                f"Source: {item['source']}\n"
                f"{item['document']}"
            )
            for item in retrieved_documents
        ]
    )

    shap_context = "\n".join(
        [
            (
                f"- {item['feature']}: "
                f"{item['shap_value']:+.4f}"
            )
            for item in shap_explanation
        ]
    )

    prompt = f"""
You are a Manufacturing AI Decision Copilot.

Analyze the machine risk using the ML prediction,
SHAP explanation, and retrieved maintenance knowledge.

USER QUESTION:
{question}

MACHINE RISK:

Failure probability:
{probability:.1%}

Risk level:
{risk_level}

SHAP EXPLANATION:
{shap_context}

RETRIEVED MAINTENANCE KNOWLEDGE:
{knowledge_context}

IMPORTANT RULES:

1. Use the retrieved maintenance knowledge as the
   primary source for maintenance guidance.

2. Do not invent maintenance procedures.

3. The ML probability represents risk, not certainty.

4. Use SHAP values to explain the important model
   contributing factors.

5. Do not repeat the retrieved documents verbatim.

6. Give a concise answer suitable for a maintenance
   engineer.

ANSWER FORMAT:

Risk assessment:
Briefly describe the machine risk.

Key factors:
List the most important factors contributing
to the prediction.

Recommended inspection:
- Action
- Action
- Action
- Action

Reason:
Briefly explain why these inspections are relevant.

Keep the response below 200 words.
"""

    # ========================================================
    # 5. GEMINI
    # ========================================================

    ai_answer = generate_answer(
        prompt
    )

    # ========================================================
    # 6. FINAL RESULT
    # ========================================================

    return {
        "failure_probability": probability,
        "risk_level": risk_level,
        "failure_prediction": prediction[
            "failure_prediction"
        ],
        "key_factors": prediction[
            "key_factors"
        ],
        "shap_explanation": shap_explanation,
        "retrieved_sources": [
            item["source"]
            for item in retrieved_documents
        ],
        "ai_answer": ai_answer
    }


if __name__ == "__main__":

    machine_data = {
        "Machine_Type": "CNC",
        "Machine_Age_Years": 10,
        "Operating_Hours": 30000,
        "Temperature_C": 75,
        "Vibration_mm_s": 5.5,
        "Pressure_bar": 6.0,
        "Motor_Current_A": 20,
        "Production_Count": 1000,
        "Production_Rate": 85,
        "Last_Maintenance_Days": 250,
        "Maintenance_Count": 8,
        "Failure_Count": 3
    }

    question = (
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    )

    result = run_copilot(
        machine_data,
        question
    )

    print("=" * 70)
    print("MANUFACTURING AI DECISION COPILOT")
    print("=" * 70)

    print("\nFAILURE PROBABILITY:")
    print(
        f"{result['failure_probability']:.1%}"
    )

    print("\nRISK LEVEL:")
    print(
        result["risk_level"]
    )

    print("\nKEY FACTORS:")

    for factor in result[
        "key_factors"
    ]:
        print(
            f"- {factor}"
        )

    print("\nRETRIEVED SOURCES:")

    for source in result[
        "retrieved_sources"
    ]:
        print(
            f"- {source}"
        )

    print("\nAI ANSWER:")
    print(
        result["ai_answer"]
    )