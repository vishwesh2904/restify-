import streamlit as st
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from utils.helper import load_model, FEATURE_COLS  # Make sure this loads both model and label_encoder

def show_admin_panel():
    st.title("🔧 Admin Panel - Feedback Management")

    feedback_file = "data/feedback.csv"

    if not os.path.exists(feedback_file):
        st.warning("⚠️ No feedback data available yet.")
        return

    df = pd.read_csv(feedback_file)

    # --- Show Data Preview ---
    st.subheader("📋 Feedback Data Preview")
    st.dataframe(df, use_container_width=True)

    # --- Summary Statistics ---
    st.subheader("📊 Feedback Summary")
    st.write("Total Feedback Entries:", len(df))

    if "Rating" in df.columns:
        avg_rating = df["Rating"].mean()
        st.write(f"Average Song Rating: {avg_rating:.2f}")

    # --- Filter Options ---
    st.subheader("🔍 Filter Feedback")
    insomnia_levels = df["Insomnia Level"].unique().tolist()
    selected_level = st.selectbox("Select Insomnia Level to Filter", ["All"] + insomnia_levels)

    if selected_level != "All":
        df = df[df["Insomnia Level"] == selected_level]
        st.dataframe(df, use_container_width=True)

    # --- Download Option ---
    st.subheader("📥 Download Filtered Feedback")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="filtered_feedback.csv", mime="text/csv")

    # --- Delete Option ---
    st.subheader("🗑️ Delete All Feedback (Admin Only)")
    if st.button("Delete All Feedback"):
        os.remove(feedback_file)
        st.success("✅ All feedback has been deleted.")

    # --- Model Training Section ---
    st.subheader("⚙️ Model Training")
    if st.button("Retrain Model"):
        from utils.helper import retrain_model_with_feedback
        with st.spinner("Retraining model, please wait..."):
            accuracy = retrain_model_with_feedback()
        st.success(f"✅ Model retrained successfully with accuracy: {accuracy * 100:.2f}%")

    # --- Visualization ---
    plot_insomnia_distribution()

    # --- Show Model Accuracy ---
    st.subheader("📏 Model Accuracy")
    show_model_accuracy()


def plot_insomnia_distribution():
    st.subheader("📈 Insomnia Level Distribution")
    try:
        df = pd.read_csv("data/insomnia_synthetic.csv")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Insomnia Level", order=["No Insomnia", "Mild", "Moderate", "Severe"], ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading data: {e}")


def show_model_accuracy():
    try:
        df = pd.read_csv("data/insomnia_synthetic.csv")
        if "Insomnia Level" not in df.columns:
            st.warning("Insomnia Level column missing.")
            return

        # Load model, label encoder, and scaler (added scaler for consistency)
        model, label_encoder, scaler = load_model()

        # Use FEATURE_COLS explicitly to select features for prediction
        X = df[FEATURE_COLS]
        
        # Apply scaling if the model was trained on scaled data
        X_scaled = scaler.transform(X)

        # Get the true labels
        y_true = label_encoder.transform(df["Insomnia Level"])

        # Model Prediction
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y_true, y_pred)

        st.write(f"✅ Model Accuracy on Stored Data: **{acc * 100:.2f}%**")

    except Exception as e:
        st.error(f"Error computing model accuracy: {e}")

