import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

from dotenv import load_dotenv
load_dotenv()

from utils.helper import load_model, get_questions
from utils.recommender import recommend_song_from_dataset

questions = get_questions()

def collect_feedback(insomnia_level, song_label):
    st.subheader("🗣️ Feedback")
    relaxed = st.radio("Did you feel relaxed?", ["Yes", "No"])
    sleep = st.radio("Did it help you sleep?", ["Yes", "No"])
    rating = st.slider("Rate the song", 1, 5)
    comments = st.text_area("Additional comments (optional)")

    feedback_entry = {
        "Insomnia Level": insomnia_level,
        "Recommended Song": song_label,
        "Felt Relaxed": relaxed,
        "Fell Asleep": sleep,
        "Rating": rating,
        "Comments": comments
    }

    feedback_file = "data/feedback.csv"
    feedback_df = pd.DataFrame([feedback_entry])

    os.makedirs("data", exist_ok=True)
    if os.path.exists(feedback_file):
        feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(feedback_file, index=False)

    st.success("✅ Thank you! Your feedback has been saved.")

def show_metrics():
    st.subheader("📊 Model Evaluation")
    df = pd.read_csv("data/insomnia_synthetic.csv")
    model, label_encoder = load_model()

    X = df[questions]
    y = label_encoder.transform(df["Insomnia Level"])
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    st.text("Classification Report:")
    st.text(classification_report(y, y_pred, target_names=label_encoder.classes_))

    st.text("Confusion Matrix:")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, cmap='Blues', ax=ax)
    st.pyplot(fig)

def plot_insomnia_distribution():
    st.subheader("📈 Insomnia Level Distribution")
    df = pd.read_csv("data/insomnia_synthetic.csv")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Insomnia Level", order=["No Insomnia", "Mild", "Moderate", "Severe"], ax=ax)
    st.pyplot(fig)

def main():
    st.title("🧠 Insomnia Detection & Smart Lullaby Recommender")
    st.write("Answer the following questions (0 = None, 4 = Very Severe):")

    user_input = []
    for q in questions:
        user_input.append(st.slider(q, 0, 4, 0))

    num_songs = st.slider("How many songs would you like to be recommended?", 1, 10, 1)

    if st.button("Predict & Recommend"):
        model, label_encoder = load_model()
        prediction = model.predict([user_input])[0]
        insomnia_level = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🧠 Predicted Insomnia Level: **{insomnia_level}**")

        labels, links = recommend_song_from_dataset(insomnia_level, num_songs)
        for i, (label, link) in enumerate(zip(labels, links), start=1):
            st.info(f"🎵 Recommended Song {i}: {label}")
            if link:
                st.markdown(f"[▶️ Listen on Spotify]({link})")

        # Collect feedback for the first recommended song only for simplicity
        collect_feedback(insomnia_level, labels[0] if labels else "No song")

    st.markdown("---")
    show_metrics()
    plot_insomnia_distribution()

if __name__ == "__main__":
    main()
