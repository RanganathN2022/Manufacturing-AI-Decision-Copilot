import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import streamlit as st

from app.copilot import run_copilot


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Manufacturing AI Decision Copilot",
    page_icon="🏭",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.title("🏭 Manufacturing AI Decision Copilot")

st.write(
    "AI-assisted predictive maintenance using "
    "Machine Learning, SHAP, RAG and Gemini."
)

st.divider()


# ---------------------------------------------------------
# MACHINE INPUT
# ---------------------------------------------------------

st.subheader("Machine Information")

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
# USER QUESTION
# ---------------------------------------------------------

st.subheader("Maintenance Question")

question = st.text_area(
    "Ask the maintenance assistant",
    value=(
        "The machine has high vibration. "
        "What should the maintenance team inspect?"
    ),
    height=100
)


# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
):

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
        "Analyzing machine condition..."
    ):

        try:

            result = run_copilot(
                machine_data,
                question,
                top_k=2
            )

            st.session_state["result"] = result

        except Exception as e:

            st.error(
                f"An error occurred: {e}"
            )


# ---------------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------------

if "result" in st.session_state:

    result = st.session_state["result"]

    st.divider()

    st.subheader("Machine Risk Assessment")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Failure Probability",
            f"{result['failure_probability']:.1%}"
        )

    with col2:

        st.metric(
            "Risk Level",
            result["risk_level"]
        )

    with col3:

        prediction_text = (
            "Failure Risk Detected"
            if result["failure_prediction"] == 1
            else "Normal"
        )

        st.metric(
            "Prediction",
            prediction_text
        )


    # -----------------------------------------------------
    # KEY FACTORS
    # -----------------------------------------------------

    st.subheader("Key Risk Factors")

    for factor in result["key_factors"]:

        st.write(
            f"• {factor}"
        )


    # -----------------------------------------------------
    # SHAP EXPLANATION
    # -----------------------------------------------------

    st.subheader("SHAP Model Explanation")

    shap_data = result[
        "shap_explanation"
    ]

    if shap_data:

        for item in shap_data:

            feature = (
                item["feature"]
                .replace(
                    "numerical__",
                    ""
                )
                .replace(
                    "categorical__",
                    ""
                )
            )

            shap_value = item[
                "shap_value"
            ]

            direction = (
                "increases risk"
                if shap_value > 0
                else "reduces risk"
            )

            st.write(
                f"**{feature}** — "
                f"{shap_value:+.4f} "
                f"({direction})"
            )


    # -----------------------------------------------------
    # RAG SOURCES
    # -----------------------------------------------------

    st.subheader(
        "Retrieved Maintenance Knowledge"
    )

    for source in result[
        "retrieved_sources"
    ]:

        st.write(
            f"📄 {source}"
        )


    # -----------------------------------------------------
    # AI ANSWER
    # -----------------------------------------------------

    st.subheader(
        "🤖 AI Maintenance Recommendation"
    )

    st.info(
        result["ai_answer"]
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Manufacturing AI Decision Copilot | "
    "Machine Learning + SHAP + RAG + Gemini"
)