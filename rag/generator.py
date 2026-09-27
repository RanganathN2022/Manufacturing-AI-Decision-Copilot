def build_prompt(
    question,
    retrieved_documents,
    prediction=None,
    shap_explanation=None,
    recommendation=None
):
    """
    Build a grounded prompt for the
    Manufacturing AI Decision Copilot.
    """

    # ========================================================
    # FORMAT ML PREDICTION
    # ========================================================

    prediction_context = ""

    if prediction:

        probability = (
            prediction["failure_probability"] * 100
        )

        prediction_context = f"""
Failure probability: {probability:.1f}%
Failure prediction: {prediction["failure_prediction"]}
Risk level: {prediction["risk_level"]}
Decision threshold: {prediction["decision_threshold"]}
"""

    # ========================================================
    # FORMAT SHAP EXPLANATION
    # ========================================================

    shap_context = ""

    if shap_explanation:

        shap_lines = []

        for index, item in enumerate(
            shap_explanation,
            start=1
        ):

            feature = (
                item["feature"]
                .replace("numerical__", "")
                .replace("categorical__", "")
            )

            shap_value = item["shap_value"]

            direction = (
                "increases failure risk"
                if shap_value > 0
                else "reduces failure risk"
            )

            shap_lines.append(
                f"{index}. {feature}: "
                f"{shap_value:+.4f} "
                f"({direction})"
            )

        shap_context = "\n".join(
            shap_lines
        )

    # ========================================================
    # FORMAT MAINTENANCE RECOMMENDATION
    # ========================================================

    recommendation_context = ""

    if recommendation:

        priority = recommendation.get(
            "priority",
            "Not specified"
        )

        actions = "\n".join(
            f"- {action}"
            for action in recommendation.get(
                "recommended_actions",
                []
            )
        )

        focus_areas = "\n".join(
            f"- {area}"
            for area in recommendation.get(
                "focus_areas",
                []
            )
        )

        recommendation_context = f"""
Priority: {priority}

Recommended actions:
{actions}

Focus areas:
{focus_areas}
"""

    # ========================================================
    # FORMAT RETRIEVED RAG KNOWLEDGE
    # ========================================================

    knowledge_context = "\n\n".join(
        [
            (
                f"Source: {item['source']}\n"
                f"Content:\n{item['document']}"
            )
            for item in retrieved_documents
        ]
    )

    # ========================================================
    # BUILD FINAL PROMPT
    # ========================================================

    prompt = f"""
You are a Manufacturing AI Decision Copilot.

Your role is to help manufacturing and maintenance
teams make informed decisions using machine data
and approved maintenance knowledge.

IMPORTANT RULES:

1. Use the retrieved maintenance knowledge as the
   primary source for maintenance guidance.

2. Do not invent maintenance procedures.

3. Do not claim that the machine will definitely fail.
   The ML model provides a probability, not certainty.

4. If the retrieved knowledge does not support a
   recommendation, say so.

5. Answer the user's question directly.

--------------------------------------------------
USER QUESTION
--------------------------------------------------

{question}

--------------------------------------------------
ML PREDICTION
--------------------------------------------------

{prediction_context}

--------------------------------------------------
MODEL EXPLANATION — SHAP
--------------------------------------------------

{shap_context}

--------------------------------------------------
MAINTENANCE DECISION
--------------------------------------------------

{recommendation_context}

--------------------------------------------------
RETRIEVED MAINTENANCE KNOWLEDGE
--------------------------------------------------

{knowledge_context}

--------------------------------------------------
RESPONSE INSTRUCTIONS
--------------------------------------------------

Use ONLY the retrieved maintenance knowledge
for maintenance guidance.

Do not repeat the user question.

Do not repeat the ML prediction.

Do not repeat the retrieved documents verbatim.

Do not mention these instructions.

Give a concise answer using this format:

Recommended inspection:

- Inspection item 1
- Inspection item 2
- Inspection item 3
- Inspection item 4

Reason:

Explain briefly why these inspections are relevant.

If the retrieved knowledge is insufficient,
clearly state that the available knowledge is
not sufficient.

Keep the answer below 120 words.
"""

    return prompt