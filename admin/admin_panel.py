import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper import load_model, FEATURE_COLS

def show_admin_panel():
    st.title("🛠️ Admin Panel")

    feedback_file = "data/feedback.csv"
    synthetic_data_file = "data/insomnia_synthetic.csv"

    # --- Feedback Section ---
    st.header("📋 User Feedback")
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)

        st.subheader("Preview")
        st.dataframe(df, use_container_width=True)

        st.markdown(f"**Total Entries:** {len(df)}")
        if "Rating" in df.columns:
            st.markdown(f"**Average Rating:** {df['Rating'].mean():.2f} ⭐")

        # Filter by Insomnia Level
        st.subheader("🔍 Filter Feedback")
        levels = df["Insomnia Level"].unique().tolist()
        selected = st.selectbox("Filter by Insomnia Level", ["All"] + levels)
        if selected != "All":
            df = df[df["Insomnia Level"] == selected]
        st.dataframe(df, use_container_width=True)

        # Download Button
        st.download_button("📥 Download Filtered Feedback", df.to_csv(index=False), "filtered_feedback.csv")

        # Delete All
        st.subheader("🗑️ Delete All Feedback")
        if st.button("Delete All Feedback"):
            os.remove(feedback_file)
            st.success("All feedback deleted.")
    else:
        st.warning("No feedback data available.")

    # # --- Retrain Model Section ---
    # st.header("🔄 Model Retraining")
    # if st.button("Retrain Model"):
    #     with st.spinner("Retraining..."):
    #         from utils.helper import retrain_model_with_feedback
    #         try:
    #             accuracy = retrain_model_with_feedback()
    #             st.success(f"✅ Model retrained successfully with accuracy: **{accuracy * 100:.2f}%**")
    #         except Exception as e:
    #             st.error(f"❌ Error retraining model: {e}")

    # --- Accuracy Display Section ---
    st.header("📏 Current Model Accuracy")
    try:
        cleaned_data_file = "data/insomnia_synthetic_cleaned.csv"
        data_file_to_use = cleaned_data_file if os.path.exists(cleaned_data_file) else synthetic_data_file

        df = pd.read_csv(data_file_to_use)

        # Validate required data
        if not set(FEATURE_COLS).issubset(df.columns):
            st.error("Required features missing from dataset.")
            return

        model_path = "models/insomnia_model.pkl"
        label_encoder_path = "models/label_encoder.pkl"
        scaler_path = "models/scaler.pkl"
        if not (os.path.exists(model_path) and os.path.exists(label_encoder_path) and os.path.exists(scaler_path)):
            st.error("Model files are missing. Please retrain the model.")
            return

        model, label_encoder, scaler = load_model()

        X = df[FEATURE_COLS]
        y = df["Insomnia Level"]

        # Convert feature columns to numeric if not already
        X = X.apply(pd.to_numeric, errors='coerce')

        # Label encoding
        y_encoded = label_encoder.transform(y)
        X_scaled = scaler.transform(X)

        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        cm = confusion_matrix(y_encoded, y_pred)

        st.success(f"✅ Current Model Accuracy: **{accuracy * 100:.2f}%**")
        st.success(f"ℹ️ F1 Score (weighted): **{f1:.4f}**")

        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Could not compute model accuracy: {e}")

