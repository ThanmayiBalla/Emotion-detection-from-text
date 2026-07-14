# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import read_logs

def render_dashboard_ui():
    st.markdown("## 📊 Analytics Dashboard")
    st.markdown("Monitor real-time system usage metrics and classification patterns.")
    
    df = read_logs()
    
    if df.empty:
        st.info("No prediction data logged yet. Run predictions to populate analytics!")
        return
        
    # High-level KPIs
    total_predictions = len(df)
    unique_users = 1 # Offline default
    most_common = df["Primary Emotion"].mode()[0] if not df.empty else "N/A"
    avg_confidence = df["Confidence"].mean() if not df.empty else 0.0
    
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Total Runs", total_predictions)
    with kpi_col2:
        st.metric("Estimated Users", unique_users)
    with kpi_col3:
        st.metric("Top Emotion", most_common)
    with kpi_col4:
        st.metric("Avg Confidence", f"{avg_confidence*100:.1f}%")
        
    st.markdown("---")
    
    # Graphs Row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Emotion Distribution")
        emotion_counts = df["Primary Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]
        fig_pie = px.pie(emotion_counts, values="Count", names="Emotion", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.subheader("Activity Timeline")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        # Bin hourly/daily
        timeline_df = df.resample('D', on='Timestamp').count()["Student Text"].reset_index()
        timeline_df.columns = ["Date", "Inferences"]
        fig_line = px.line(timeline_df, x="Date", y="Inferences", markers=True,
                           title="Daily Application Inferences")
        st.plotly_chart(fig_line, use_container_width=True)
        
    st.markdown("---")
    
    # Model Comparison Panel
    st.subheader("🤖 Model Decision Tracking")
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        # Cross-Tabulation matrix
        st.markdown("**BiLSTM vs. BERT Model Concordance Matrix**")
        cross_tab = pd.crosstab(df["BiLSTM Prediction"], df["BERT Prediction"])
        st.dataframe(cross_tab, use_container_width=True)
        
    with comp_col2:
        # Average Confidence level of top classes
        st.markdown("**Mean Model Confidence by Primary Emotion**")
        avg_conf_group = df.groupby("Primary Emotion")["Confidence"].mean().reset_index()
        fig_bar = px.bar(avg_conf_group, x="Primary Emotion", y="Confidence", 
                         labels={"Confidence": "Average Confidence Score"},
                         color="Primary Emotion")
        st.plotly_chart(fig_bar, use_container_width=True)