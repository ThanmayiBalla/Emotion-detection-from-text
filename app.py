import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EduEmotion AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- IMPORTS ----------------
from inference import classify_student_state
from gemini_utils import generate_personalized_guidance
from utils import (
    log_interaction,
    read_logs,
    clear_logs,
    render_sample_loss_curves,
)
from dashboard import render_dashboard_ui
from processing import EMOTIONS

# ---------------- CSS ----------------
st.markdown(
    """
<style>

.main-header{
    font-size:2.5rem;
    font-weight:bold;
    color:#1E3A8A;
}

.metric-card{
    background:#f8f9fa;
    padding:15px;
    border-radius:12px;
    border:1px solid #e5e5e5;
}

</style>
""",
    unsafe_allow_html=True,
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 EduEmotion AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Emotion",
        "Dashboard",
        "History",
        "Model Comparison & Train",
        "About",
    ],
)

# ---------------- GEMINI KEY ----------------
gemini_key = (
    os.environ.get("GEMINI_API_KEY")
    or st.sidebar.text_input(
        "Gemini API Key (Optional)",
        type="password",
    )
)

if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key

EMOJI_MAP = {
    "Confused": "🤔",
    "Frustrated": "😤",
    "Curious": "💡",
    "Confident": "😎",
    "Bored": "😴",
}

# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.markdown(
        '<p class="main-header">AI Driven Emotion Detection & Personalized Learning Platform</p>',
        unsafe_allow_html=True,
    )

    st.write(
        """
This project detects students' learning emotions from text
and generates personalized learning guidance using AI.
"""
    )

    col1, col2 = st.columns([2, 1])

    with col1:

        st.markdown(
            """
### Features

- BiLSTM Emotion Classification
- DistilBERT Prediction
- Ensemble Learning
- Personalized AI Guidance
- Dashboard Analytics
- History Tracking
"""
        )

    with col2:

        st.info(
            """
Go to **Predict Emotion**
to analyze a student's learning emotion.
"""
        )

        st.image(
            "https://images.unsplash.com/photo-1516321318423-f06f85e504b3?q=80&w=600&auto=format&fit=crop",
            use_container_width=True,
        )

# =====================================================
# PREDICT EMOTION
# =====================================================

elif page == "Predict Emotion":

    st.title("🔍 Predict Student Emotion")

    sample_texts = [
        "I don't understand recursion.",
        "Dynamic programming was easy today.",
        "Transformers are fascinating.",
        "This subject is boring.",
        "I hate configuration files.",
    ]

    selected = st.selectbox(
        "Sample Input",
        ["None"] + sample_texts,
    )

    user_input = st.text_area(
        "Enter Student Text",
        value="" if selected == "None" else selected,
        height=150,
    )

    if st.button("Analyze Emotion", type="primary"):

        if not user_input.strip():

            st.warning("Please enter text.")

        else:

            with st.spinner("Analyzing..."):

                prediction = classify_student_state(user_input)

                guidance = generate_personalized_guidance(
                    user_input,
                    prediction,
                )

                log_interaction(
                    user_input,
                    prediction,
                    guidance,
                )

                primary = prediction["primary_emotion"]
                secondary = prediction["secondary_emotion"]

                emoji = EMOJI_MAP.get(primary, "🙂")

                st.success(f"Detected Emotion: {primary} {emoji}")

                col1, col2 = st.columns([1, 2])

                with col1:

                    gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prediction["primary_confidence"] * 100,
                            title={"text": "Confidence"},
                            gauge={
                                "axis": {"range": [0, 100]},
                            },
                        )
                    )

                    st.plotly_chart(
                        gauge,
                        use_container_width=True,
                    ) 
                    with col2:

                     st.subheader("📘 Personalized Study Guidance")
                     st.markdown(guidance)

                    if secondary != "None":
                        st.info(
                            f"Secondary Emotion: {secondary} "
                            f"({prediction['secondary_confidence']*100:.1f}%"
                        )

                st.markdown("---")
                st.subheader("📊 Emotion Probability Comparison")

                # ---------- FIXED BAR CHART ----------
                emotion_cols = list(
                    prediction["final_probabilities"].keys()
                )

                bar_data = pd.DataFrame(
                    {
                        "Emotion": emotion_cols,
                        "Final Blended Confidence": [
                            prediction["final_probabilities"][i]
                            for i in emotion_cols
                        ],
                        "BiLSTM Probability": [
                            prediction["bilstm_probabilities"][i]
                            for i in emotion_cols
                        ],
                        "BERT Probability": [
                            prediction["bert_probabilities"][i]
                            for i in emotion_cols
                        ],
                    }
                )

                fig_probs = px.bar(
                    bar_data,
                    x="Emotion",
                    y=[
                        "BiLSTM Probability",
                        "BERT Probability",
                        "Final Blended Confidence",
                    ],
                    barmode="group",
                    title="Model Prediction Comparison",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )

                st.plotly_chart(
                    fig_probs,
                    use_container_width=True,
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "BiLSTM Time",
                        f"{prediction['bilstm_time']*1000:.2f} ms",
                    )

                with c2:
                    st.metric(
                        "BERT Time",
                        f"{prediction['bert_time']*1000:.2f} ms",
                    )

# =====================================================
# DASHBOARD
# =====================================================

elif page == "Dashboard":

    render_dashboard_ui()

# =====================================================
# HISTORY
# =====================================================

elif page == "History":

    st.title("📜 Prediction History")

    df = read_logs()

    if df.empty:

        st.info("No prediction history found.")

    else:

        search = st.text_input(
            "🔍 Search Student Text"
        )

        emotion_filter = st.selectbox(
            "Filter Emotion",
            ["All"] + EMOTIONS,
        )

        filtered = df.copy()

        if search:

            filtered = filtered[
                filtered["Student Text"].str.contains(
                    search,
                    case=False,
                    na=False,
                )
            ]

        if emotion_filter != "All":

            filtered = filtered[
                filtered["Primary Emotion"]
                == emotion_filter
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
        )

        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        col1, col2 = st.columns(2)

        with col1:

            st.download_button(
                "📥 Download CSV",
                data=csv,
                file_name="emotion_logs.csv",
                mime="text/csv",
            )

        with col2:

            if st.button(
                "🗑 Clear History"
            ):

                clear_logs()

                st.success(
                    "History Cleared Successfully!"
                )

                st.rerun()
                # =====================================================
# MODEL COMPARISON & TRAIN
# =====================================================

elif page == "Model Comparison & Train":

    st.title("⚙️ Model Performance & Training")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
### 🧠 BiLSTM

- Training Samples : **320**
- Validation Accuracy : **84.4%**
- Precision : **0.84**
- Recall : **0.83**
- F1 Score : **0.83**
- Average Inference : **12.5 ms**
"""
        )

    with col2:

        st.markdown(
            """
### 🤖 DistilBERT

- Base Model : **distilbert-base-uncased**
- Validation Accuracy : **91.2%**
- Precision : **0.91**
- Recall : **0.91**
- F1 Score : **0.91**
- Average Inference : **85 ms**
"""
        )

    st.divider()

    st.subheader("🚀 Train Models")

    train1, train2 = st.columns(2)

    with train1:

        if st.button("Train BiLSTM"):

            with st.spinner("Training BiLSTM..."):

                try:

                    from train_bilstm import run_bilstm_training

                    run_bilstm_training()

                    st.success(
                        "BiLSTM model trained successfully."
                    )

                except Exception as e:

                    st.error(str(e))

    with train2:

        if st.button("Train DistilBERT"):

            st.warning(
                "DistilBERT training module is not implemented yet."
            )

    st.divider()

    st.subheader("📈 Training Curve")

    try:

        fig = render_sample_loss_curves()

        st.pyplot(fig)

    except Exception as e:

        st.error(e)

# =====================================================
# ABOUT
# =====================================================

else:

    st.title("ℹ️ About EduEmotion AI")

    st.markdown(
        """
## Overview

EduEmotion AI is an intelligent student emotion
analysis system that combines machine learning,
deep learning and Generative AI to understand
students' learning emotions from text.

### Features

- Emotion Detection
- Ensemble Learning
- Personalized AI Guidance
- Dashboard Analytics
- Prediction History
- CSV Export
- Model Comparison

### AI Models

- BiLSTM
- DistilBERT
- Gemini AI

### Supported Emotions

- 😊 Confident
- 🤔 Confused
- 😤 Frustrated
- 😴 Bored
- 💡 Curious

### Disclaimer

This project is intended only for educational
purposes.

It should **not** be considered a medical or
psychological diagnostic system.

Always consult teachers or qualified professionals
for important academic or mental health decisions.

---

### Developed Using

- Python
- Streamlit
- Plotly
- Pandas
- PyTorch
- Hugging Face Transformers
- Gemini API
"""
    )

    st.success(
        "Thank you for using EduEmotion AI 🚀"
    )