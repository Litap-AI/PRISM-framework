import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import plotly.graph_objects as go

from tensorflow.keras.models import load_model

from prism_rules import (
    calculate_prism_score,
    classify_product,
    generate_recommendation
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PRISM AI Framework",
    page_icon="🚀",
    layout="wide"
)

# =========================================================
# BACKGROUND FUNCTION
# =========================================================

def set_background(image_file):

    with open(image_file, "rb") as image:
        encoded = base64.b64encode(
            image.read()
        ).decode()

    background_css = f"""
    <style>

    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        background-attachment: fixed;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;

        background: rgba(0, 0, 0, 0.92);
        backdrop-filter: blur(3px);

        z-index: -1;
    }}

    h1, h2, h3, h4, h5, h6, p, div, label {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(10, 10, 20, 0.88);
    }}

    .stMetric {{
        background-color: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 12px;
    }}

    .block-container {{
        padding-top: 2rem;
    }}

    </style>
    """

    st.markdown(
        background_css,
        unsafe_allow_html=True
    )

# =========================================================
# APPLY BACKGROUND
# =========================================================

set_background("assets/assets/prism_bg.png")

# =========================================================
# LOAD MODEL + ARTIFACTS
# =========================================================

model = load_model("models/prism_ann.keras")

scaler = joblib.load("models/scaler.pkl")

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

# =========================================================
# TITLE
# =========================================================

st.title("🚀 PRISM AI Decision Framework")

st.markdown("""
### Intelligent Product Evaluation and Prioritization System
Analyze products using AI-powered PRISM scoring and ANN prediction.
""")

st.divider()

# =========================================================
# SIDEBAR INPUTS
# =========================================================

st.sidebar.title("📊 Product Parameters")

performance = st.sidebar.slider(
    "Performance",
    1,
    10,
    5
)

relevance = st.sidebar.slider(
    "Relevance",
    1,
    10,
    5
)

innovation = st.sidebar.slider(
    "Innovation",
    1,
    10,
    5
)

scalability = st.sidebar.slider(
    "Scalability",
    1,
    10,
    5
)

monetization = st.sidebar.slider(
    "Monetization",
    1,
    10,
    5
)

# =========================================================
# CSV UPLOAD SECTION
# =========================================================

st.sidebar.divider()

st.sidebar.subheader("📂 Upload Product CSV")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================================================
# CALCULATIONS
# =========================================================

score = calculate_prism_score(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
)

classification = classify_product(score)

recommendations = generate_recommendation(
    performance,
    relevance,
    innovation,
    scalability,
    monetization
)

# =========================================================
# ANN PREDICTION
# =========================================================

input_data = pd.DataFrame([
    {
        "performance": performance,
        "relevance": relevance,
        "innovation": innovation,
        "scalability": scalability,
        "monetization": monetization
    }
])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_scaled)

predicted_class = prediction.argmax(axis=1)

predicted_label = label_encoder.inverse_transform(
    predicted_class
)[0]

# =========================================================
# MAIN METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="📈 PRISM Score",
        value=round(score, 2)
    )

with col2:
    st.metric(
        label="🧠 ANN Prediction",
        value=predicted_label
    )

with col3:
    st.metric(
        label="📌 Classification",
        value=classification
    )

st.divider()

# =========================================================
# PRISM PARAMETERS DISPLAY
# =========================================================

st.subheader("📊 PRISM Parameter Scores")

param_col1, param_col2, param_col3, param_col4, param_col5 = st.columns(5)

with param_col1:
    st.metric("Performance", performance)

with param_col2:
    st.metric("Relevance", relevance)

with param_col3:
    st.metric("Innovation", innovation)

with param_col4:
    st.metric("Scalability", scalability)

with param_col5:
    st.metric("Monetization", monetization)

# =========================================================
# RADAR CHART
# =========================================================

st.divider()

st.subheader("🕸️ PRISM Radar Analysis")

categories = [
    "Performance",
    "Relevance",
    "Innovation",
    "Scalability",
    "Monetization"
]

values = [
    performance,
    relevance,
    innovation,
    scalability,
    monetization
]

# Close radar polygon
values += values[:1]
categories += categories[:1]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=categories,
    fill='toself',
    name='PRISM Analysis'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )
    ),
    showlegend=False,
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig,
    use_container_width=True
)
# =========================================================
# FEATURE CONTRIBUTION ANALYSIS
# =========================================================

st.divider()

st.subheader("🧠 PRISM Contribution Analysis")

feature_scores = {
    "Performance": performance,
    "Relevance": relevance,
    "Innovation": innovation,
    "Scalability": scalability,
    "Monetization": monetization
}

contribution_df = pd.DataFrame({
    "Feature": list(feature_scores.keys()),
    "Score": list(feature_scores.values())
})

# Sort descending
contribution_df = contribution_df.sort_values(
    by="Score",
    ascending=False
)

st.dataframe(contribution_df)

# Plot Bar Chart
fig_bar = go.Figure()

fig_bar.add_trace(go.Bar(
    x=contribution_df["Feature"],
    y=contribution_df["Score"]
))

fig_bar.update_layout(
    title="Feature Contribution Strength",
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)


# =========================================================
# RECOMMENDATIONS
# =========================================================

st.divider()

st.subheader("💡 Strategic Recommendations")

for rec in recommendations:
    st.success(rec)

# =========================================================
# ANN CONFIDENCE SCORES
# =========================================================

st.divider()

st.subheader("🤖 ANN Confidence Scores")

labels = label_encoder.classes_

for i, label in enumerate(labels):

    probability = prediction[0][i]

    st.progress(float(probability))

    st.write(
        f"**{label}** : {probability:.4f}"
    )

# =========================================================
# BATCH PRODUCT ANALYSIS
# =========================================================

if uploaded_file is not None:

    st.divider()

    st.header("📊 Batch Product Analysis")

    # Load CSV
    batch_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(batch_df)

    # Required features
    feature_columns = [
        "performance",
        "relevance",
        "innovation",
        "scalability",
        "monetization"
    ]

    # Scale features
    scaled_features = scaler.transform(
        batch_df[feature_columns]
    )

    # ANN Predictions
    predictions = model.predict(
        scaled_features
    )

    predicted_classes = predictions.argmax(axis=1)

    predicted_labels = label_encoder.inverse_transform(
        predicted_classes
    )

    # PRISM Scores
    prism_scores = []

    for _, row in batch_df.iterrows():

        score = calculate_prism_score(
            row["performance"],
            row["relevance"],
            row["innovation"],
            row["scalability"],
            row["monetization"]
        )

        prism_scores.append(score)

    # Add Results
    batch_df["PRISM_Score"] = prism_scores

    batch_df["ANN_Prediction"] = predicted_labels

    # Ranking
    batch_df = batch_df.sort_values(
        by="PRISM_Score",
        ascending=False
    )

    st.subheader("🏆 Ranked Product Analysis")

    st.dataframe(batch_df)

    # Download results
    csv = batch_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="prism_analysis_results.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown("""
### PRISM Framework

| Component | Meaning |
|---|---|
| **P** | Performance |
| **R** | Relevance |
| **I** | Innovation |
| **S** | Scalability |
| **M** | Monetization |

---

### AI-Driven Product Evaluation Framework

Built using:
- Python
- TensorFlow/Keras
- Streamlit
- Plotly
- Scikit-learn

🚀 Intelligent Decisions. High Impact Products.
""")
