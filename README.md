# 🏭 Manufacturing AI Decision Copilot

### AI-Powered Predictive Maintenance and Explainable Decision Support

An intelligent manufacturing maintenance assistant that combines **Machine Learning, SHAP Explainability, Retrieval-Augmented Generation (RAG), and Google Gemini** to assess machine failure risk and provide context-aware maintenance recommendations.

<p align="center">
  <a href="https://ranganathn2022-manufacturing-ai-decisio-appstreamlit-app-vkwjlf.streamlit.app/">
    <strong>🚀 Live Application</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/RanganathN2022/Manufacturing-AI-Decision-Copilot">
    <strong>💻 GitHub Repository</strong>
  </a>
</p>

---

## 📌 Project Overview

Unexpected equipment failures can lead to production downtime, maintenance costs, and operational disruptions.

The Manufacturing AI Decision Copilot demonstrates how machine learning and generative AI can work together to support predictive maintenance decisions.

Users can enter machine operating parameters, estimate failure probability, understand the factors influencing the prediction, retrieve relevant maintenance knowledge, and receive AI-generated inspection recommendations.

## 🎯 Key Features

### 1. Predictive Maintenance Using Machine Learning
- Predicts machine failure risk using a trained Random Forest model.
- Estimates failure probability.
- Classifies machine risk.
- Displays the model's binary prediction.

### 2. Explainable AI Using SHAP
- Explains the factors influencing model predictions.
- Identifies important machine parameters.
- Shows positive and negative feature contributions.
- Improves transparency and interpretability.

### 3. Retrieval-Augmented Generation (RAG)
- Retrieves relevant information from maintenance knowledge documents.
- Uses Sentence Transformers for semantic embeddings.
- Uses ChromaDB as the vector database.
- Provides contextual knowledge to support AI-generated responses.

### 4. AI Maintenance Assistant Using Google Gemini
- Accepts natural-language maintenance questions.
- Uses retrieved maintenance knowledge as context.
- Generates inspection recommendations.
- Connects predictive insights with maintenance guidance.

### 5. Interactive Streamlit Dashboard
- Machine operating parameter inputs.
- Failure probability indicator.
- Risk classification and prediction.
- SHAP feature contribution chart.
- Retrieved knowledge sources.
- AI-generated maintenance recommendations.

---

## 🧠 System Architecture

The application follows this workflow:

```text
       Machine Operating Parameters
                   |
                   v
          Streamlit Dashboard
                   |
                   v
          Machine Learning Model
             Random Forest
                   |
          +--------+--------+
          |                 |
          v                 v
    Failure Risk       SHAP Explainability
    Probability        Feature Contributions
          |                 |
          +--------+--------+
                   |
                   v
          Maintenance Assistant
                   |
                   v
             RAG Pipeline
                   |
          +--------+--------+
          |                 |
          v                 v
     Sentence          ChromaDB
    Transformers      Vector Search
          |                 |
          +--------+--------+
                   |
                   v
          Retrieved Knowledge
                   |
                   v
             Google Gemini
                   |
                   v
       AI Maintenance Recommendation
```

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Predictive Model | Random Forest |
| Explainable AI | SHAP |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Generative AI | Google Gemini |
| Application Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Deployment | Streamlit Community Cloud |
| Version Control | Git and GitHub |

## 📂 Project Structure

```text
Manufacturing-AI-Decision-Copilot/
│
├── app/
│   ├── streamlit_app.py
│   ├── copilot.py
│   └── predict.py
│
├── data/
│   └── raw/
│
├── evaluation/
│
├── knowledge/
│   ├── maintenance_sop.txt
│   ├── temperature_guide.txt
│   └── vibration_guide.txt
│
├── models/
│   ├── failure_prediction_model.joblib
│   └── model_config.json
│
├── notebooks/
│
├── rag/
│   ├── ingest.py
│   ├── retrieve.py
│   └── gemini.py
│
├── requirements.txt
├── runtime.txt
├── project_structure.txt
└── README.md
```

*Note: The structure above highlights the main project components; check the repository for the complete current file list.*

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.12 (recommended, matching the deployment runtime configuration).
- Git.
- Google Gemini API key.

### Step 1: Clone the repository

```bash
git clone https://github.com/RanganathN2022/Manufacturing-AI-Decision-Copilot.git

cd Manufacturing-AI-Decision-Copilot
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment.

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 4: Configure Gemini API key

Create a `.env` file if your application configuration loads environment variables from it:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Alternatively, configure the key using Streamlit secrets:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

Never commit API keys, `.env` files, or other credentials to GitHub.

### Step 5: Run the application

```bash
streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal.

## 🚀 Live Demo

Try the deployed application:

**[Manufacturing AI Decision Copilot – Live Demo](https://ranganathn2022-manufacturing-ai-decisio-appstreamlit-app-vkwjlf.streamlit.app/)**

Example maintenance question:

> The machine has high vibration. What should the maintenance team inspect?

The application evaluates the machine condition, displays risk indicators, retrieves relevant maintenance information, and generates an AI-assisted recommendation.

## 🔍 Example Application Workflow

1. Enter machine information such as machine age, operating hours, temperature, vibration and maintenance history.
2. Submit the information for analysis.
3. View estimated failure probability and risk classification.
4. Examine SHAP explanations to understand influential features.
5. Review the retrieved maintenance knowledge.
6. Read the Gemini-generated maintenance recommendation.

## 🔐 Responsible Use

This project is an AI-assisted decision-support prototype.

- Predictions are estimates from a trained machine learning model.
- SHAP explanations describe model behaviour, not necessarily physical causation.
- AI-generated recommendations should be checked against equipment manuals, maintenance procedures and applicable safety requirements.
- The application should not replace qualified maintenance personnel or safety inspections.

## 🔮 Future Enhancements

- Real-time IoT sensor integration.
- Time-series machine health monitoring.
- Predictive remaining useful life estimation.
- Maintenance scheduling and alert notifications.
- Advanced model evaluation and monitoring.
- Multi-machine fleet analytics.
- Improved visual analytics and reporting.

## 👨‍💻 Author

**Ranganath N.**

AI/ML | Generative AI | Agentic AI | Manufacturing Applications

GitHub: [RanganathN2022](https://github.com/RanganathN2022)

---

⭐ If you find this project interesting, consider starring the repository.

**Built to explore the practical integration of Machine Learning, Explainable AI, RAG and Generative AI in manufacturing maintenance.**
