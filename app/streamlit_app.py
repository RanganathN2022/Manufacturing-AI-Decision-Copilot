
import sys
from pathlib import Path

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from app.copilot import run_copilot


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Manufacturing AI Decision Copilot",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
# CUSTOM STYLING
# ---------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 750;
        color: #183153;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-heading {
        color: #183153;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    .info-card {
        background-color: #f1f5f9;
        padding: 16px;
        border-radius: 10px;
        border-left: 4px solid #2563eb;
        margin-bottom: 12px;
    }

    .footer {
        color: #64748b;
        text-align: center;
        font-size: 0.8rem;
        padding-top: 20px;
    }

    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 16px;
        border-radius: 10px;
    }

    div.stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 45px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-header">🏭 Manufacturing AI Decision Copilot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-header">'
    'Intelligent predictive maintenance powered by '
    'Machine Learning, SHAP, RAG and Google Gemini.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------------------------------
# MACHINE INFORMATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">⚙️ Machine Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["CNC", "Lathe", "Milling", "Drilling"]
    )

    machine_age = st.number_input(
        "Machine Age (Years)",
        min_value=0,
        max_value=50,
        value=10
    )

    operating_hours = st.number_input(
        "Operating Hours",
        min_value=0,
        max_value=100000,
        value=30000
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=200.0,
        value=75.0
    )


with col2:
    vibration = st.number_input(
        "Vibration (mm/s)",
        min_value=0.0,
        max_value=50.0,
        value=5.5
    )

    pressure = st.number_input(
        "Pressure (bar)",
        min_value=0.0,
        max_value=50.0,
        value=6.0
    )

    motor_current = st.number_input(
        "Motor Current (A)",
        min_value=0.0,
        max_value=200.0,
        value=20.0
    )

    production_count = st.number_input(
        "Production Count",
        min_value=0,
        max_value=100000,
        value=1000
    )


with col3:
    production_rate = st.number_input(
        "Production Rate",
        min_value=0.0,
        max_value=100.0,
        value=85.0
    )

    last_maintenance = st.number_input(
        "Days Since Last Maintenance",
        min_value=0,
        max_value=2000,
        value=250
    )

    maintenance_count = st.number_input(
        "Maintenance Count",
        min_value=0,
        max_value=100,
        value=8
    )

    failure_count = st.number_input(
        "Previous Failure Count",
        min_value=0,
        max_value=50,
        value=3
    )


# ---------------------------------------------------------
# MAINTENANCE QUESTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">💬 Maintenance Assistant</div>',
    unsafe_allow_html=True
)

question = st.text_area(
    "Ask a question about the machine condition",
    value=(
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    ),
    height=90
)


# ---------------------------------------------------------
# MACHINE ANALYSIS
# ---------------------------------------------------------

analyze = st.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
)

if analyze:

    machine_data = {
        "Machine_Type": machine_type,
        "Machine_Age_Years": machine_age,
        "Operating_Hours": operating_hours,
        "Temperature_C": temperature,
        "Vibration_mm_s": vibration,
        "Pressure_bar": pressure,
        "Motor_Current_A": motor_current,
        "Production_Count": production_count,
        "Production_Rate": production_rate,
        "Last_Maintenance_Days": last_maintenance,
        "Maintenance_Count": maintenance_count,
        "Failure_Count": failure_count
    }

    with st.spinner(
        "Analyzing machine condition using ML, SHAP, RAG and Gemini..."
    ):
        try:
            result = run_copilot(
                machine_data,
                question,
                top_k=2
            )

            st.session_state["result"] = result
            st.session_state["analysis_error"] = None

        except Exception as e:
            st.session_state["analysis_error"] = str(e)
            st.session_state.pop("result", None)


# ---------------------------------------------------------
# ERROR DISPLAY
# ---------------------------------------------------------

if st.session_state.get("analysis_error"):
    st.error(
        f"Analysis failed: {st.session_state['analysis_error']}"
    )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.markdown(
        '<div class="section-heading">📊 Machine Risk Assessment</div>',
        unsafe_allow_html=True
    )

    probability = float(result["failure_probability"])
    risk_level = str(result["risk_level"])

    prediction = result["failure_prediction"]

    prediction_text = (
        "Failure Risk Detected"
        if prediction == 1
        else "Normal"
    )

    # Three KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Failure Probability",
            f"{probability:.1%}"
        )

    with col2:
        st.metric(
            "Risk Classification",
            risk_level
        )

    with col3:
        st.metric(
            "Model Prediction",
            prediction_text
        )

    # Probability indicator
    st.markdown("**Failure Probability Indicator**")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    threshold = result.get("decision_threshold")

    if threshold is not None:
        st.caption(
            f"Classification threshold: {float(threshold):.0%}. "
            "The risk category and binary prediction are separate outputs."
        )
    else:
        st.caption(
            "The probability indicator represents the model's estimated "
            "failure probability. Risk classification and binary prediction "
            "are separate outputs."
        )

    # -----------------------------------------------------
    # KEY RISK FACTORS
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">⚠️ Key Risk Factors</div>',
        unsafe_allow_html=True
    )

    factors = result.get("key_factors", [])

    if factors:
        factor_columns = st.columns(2)

        for index, factor in enumerate(factors):
            with factor_columns[index % 2]:
                st.markdown(
                    f'<div class="info-card">⚠️ {factor}</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No key risk factors were returned.")


    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">🧠 SHAP Model Explanation</div>',
        unsafe_allow_html=True
    )

    shap_data = result.get("shap_explanation", [])

    if shap_data:

        shap_rows = []

        for item in shap_data:

            feature = (
                str(item["feature"])
                .replace("numerical__", "")
                .replace("categorical__", "")
                .replace("_", " ")
            )

            shap_value = float(item["shap_value"])

            direction = (
                "Increases risk"
                if shap_value > 0
                else "Reduces risk"
                if shap_value < 0
                else "No contribution"
            )

            shap_rows.append({
                "Feature": feature,
                "SHAP Value": shap_value,
                "Effect": direction
            })

        shap_df = pd.DataFrame(shap_rows)

        st.caption(
            "Positive contributions increase the model output; "
            "negative contributions reduce it."
        )

        st.bar_chart(
            shap_df.set_index("Feature")[["SHAP Value"]],
            horizontal=True
        )

        with st.expander("View SHAP values and details"):

            display_df = shap_df.copy()

            display_df["SHAP Value"] = display_df[
                "SHAP Value"
            ].map(lambda x: f"{x:+.4f}")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

    else:
        st.info("SHAP explanation is not available.")


    # -----------------------------------------------------
    # RETRIEVED KNOWLEDGE
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">📚 Retrieved Maintenance Knowledge</div>',
        unsafe_allow_html=True
    )

    sources = result.get("retrieved_sources", [])

    if sources:

        for source in sources:
            st.markdown(f"- 📄 `{source}`")

    else:
        st.info("No maintenance documents were retrieved.")


    # -----------------------------------------------------
    # GEMINI RECOMMENDATION
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-heading">🤖 AI Maintenance Recommendation</div>',
        unsafe_allow_html=True
    )

    ai_answer = result.get(
        "ai_answer",
        "No recommendation was generated."
    )

    st.markdown(
        ai_answer
    )

    st.caption(
        "AI-generated decision support. Maintenance personnel should "
        "validate recommendations against equipment manuals, site procedures "
        "and safety requirements before taking action."
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    'Manufacturing AI Decision Copilot | '
    'Machine Learning • SHAP • RAG • Google Gemini'
    '<br>Predictive maintenance and explainable AI decision support'
    '</div>',
    unsafe_allow_html=True
)
