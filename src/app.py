import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import plotly.graph_objects as go
import plotly.express as px

from tensorflow.keras.models import load_model

from src.prism_rules import (
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
        background: rgba(0, 0, 0, 0.88);
        z-index: -1;
    }}

    h1, h2, h3, h4, h5, h6, p, div, label {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] {{
        background: rgba(10, 10, 20, 0.90);
    }}

    .stMetric {{
        background-color: rgba(255,255,255,0.08);
        padding: 15px;
        border-radius: 12px;
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
# LOAD MODEL
# =========================================================

try:
    model = load_model("models/prism_ann.keras")
    scaler = joblib.load("models/scaler.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")
    model_loaded = True

except Exception as e:
    st.warning(f"ANN model could not be loaded: {e}")
    model_loaded = False


# =========================================================
# TITLE
# =========================================================

st.title("🚀 PRISM AI Decision Framework")

st.caption(
    "AI-powered product ranking and strategic prioritization framework"
)

st.divider()

# ============================================================
# STRATEGIC PRIORITY WEIGHTS
# ============================================================

st.sidebar.header("🎯 Strategic Priority Weights")

performance_weight = st.sidebar.slider(
    "Performance Weight",
    0,
    10,
    5
)

relevance_weight = st.sidebar.slider(
    "Relevance Weight",
    0,
    10,
    5
)

innovation_weight = st.sidebar.slider(
    "Innovation Weight",
    0,
    10,
    5
)

scalability_weight = st.sidebar.slider(
    "Scalability Weight",
    0,
    10,
    5
)

monetization_weight = st.sidebar.slider(
    "Monetization Weight",
    0,
    10,
    5
)

# =========================================================
# TABS
# =========================================================

tab1, tab2 = st.tabs([
    "🎯 Strategic Product Search",
    "📂 CSV Dataset Analysis"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.subheader("🎯 Strategic Preference Configuration")

    st.caption(
        "Set your ideal product profile from 0 (low priority) to 10 (high priority)"
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        performance = st.slider(
            "Performance",
            0,
            10,
            5
        )

    with col2:
        relevance = st.slider(
            "Relevance",
            0,
            10,
            5
        )

    with col3:
        innovation = st.slider(
            "Innovation",
            0,
            10,
            5
        )

    with col4:
        scalability = st.slider(
            "Scalability",
            0,
            10,
            5
        )

    with col5:
        monetization = st.slider(
            "Monetization",
            0,
            10,
            5
        )
# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("📂 CSV Dataset Upload")

    st.info(
        "Upload product metadata CSV to generate strategic ranking and ANN predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Product Metadata CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.success("✅ CSV Uploaded Successfully")

        # READ CSV
        df = pd.read_csv(uploaded_file)

        # SHOW DATA
        st.subheader("📊 Dataset Preview")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # REQUIRED COLUMNS
        required_columns = [
            "performance",
            "relevance",
            "innovation",
            "scalability",
            "monetization"
        ]

        missing_cols = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_cols:

            st.error(
                f"❌ Missing columns: {missing_cols}"
            )

        else:

            if st.button("🚀 Generate PRISM Analysis"):

             with st.spinner("Running PRISM Strategic Intelligence Engine..."):

        # ============================================
        # CUSTOM WEIGHTED SCORING
        # ============================================

                df ["PRISM_score"] = (
                     df["performance"] * performance_weight +
                     df["relevance"] * relevance_weight +
                     df["innovation"] * innovation_weight +
                     df["scalability"] * scalability_weight +
                     df["monetization"] * monetization_weight
                )

        # ============================================
        # ANN PREDICTION
        # ============================================

        required_columns = [
            "performance",
            "relevance",
            "innovation",
            "scalability",
            "monetization"
        ]

        scaled_input = scaler.transform(
            df[required_columns]
        )

        predictions = model.predict(
            scaled_input,
            verbose=0
        )

        predicted_classes = predictions.argmax(
            axis=1
        )

        predicted_labels = (
            label_encoder.inverse_transform(
                predicted_classes
            )
        )

        df["ANN_Prediction"] = predicted_labels

        # ============================================
        # SORT RESULTS
        # ============================================

        ranked_df = df.sort_values(
            by="PRISM_score",
            ascending=False
        )

        top_10 = ranked_df.head(10)

        # ============================================
        # RESULTS
        # ============================================

        st.success("✅ PRISM Analysis Completed")

        st.subheader("🏆 Top 10 Strategic Product Matches")

        st.dataframe(
            top_10,
            use_container_width=True
        )

        # ============================================
        # CHART
        # ============================================

        fig = px.bar(
            top_10,
            x="product_name",
            y="PRISM_score",
            color="ANN_Prediction",
            title="Top Ranked Strategic Products"
        )

        fig.update_layout(
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ============================================
        # DOWNLOAD
        # ============================================

        csv = ranked_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download Ranked Results",
            data=csv,
            file_name="prism_ranked_results.csv",
            mime="text/csv"
        )

# =========================================================
# PRISM CALCULATIONS
# =========================================================

st.divider()

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

if model_loaded:

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_class = prediction.argmax(axis=1)

    predicted_label = label_encoder.inverse_transform(
        predicted_class
    )[0]

    st.success(f"🤖 ANN Prediction: {predicted_label}")

else:

    st.info("⚠️ ANN model unavailable in deployment mode.")

# =========================================================
# MAIN METRICS
# =========================================================

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "📈 PRISM Score",
        round(score, 2)
    )

with m2:
    st.metric(
        "🤖 ANN Prediction",
        predicted_label
    )

with m3:
    st.metric(
        "📌 Classification",
        classification
    )

# =========================================================
# RADAR CHART
# =========================================================

st.divider()

st.subheader("🕸️ Strategic Radar Analysis")

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

values += values[:1]
categories += categories[:1]

radar_fig = go.Figure()

radar_fig.add_trace(
    go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        name="PRISM"
    )
)

radar_fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 10]
        )
    ),
    template="plotly_dark",
    height=500
)

st.plotly_chart(
    radar_fig,
    use_container_width=True
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
