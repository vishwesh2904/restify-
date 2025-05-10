import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# --- Auth imports ---
from auth.auth_manager import create_user_table, signup_user, login_user
from utils.helper import load_model, get_questions
from utils.recommender import recommend_song_from_dataset

# --- Initialize DB ---
create_user_table()

# --- Login Session Management ---
import streamlit as st
from auth.auth_manager import signup_user, login_user

# --- Login Session Management ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    st.sidebar.title("🔐 Authentication")
    option = st.sidebar.radio("Login/Signup", ("Login", "Signup"))

    if option == "Signup":
        with st.sidebar.form("signup_form"):
            username = st.text_input("Choose a Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")
            if submit:
                if signup_user(username, email, password):
                    st.success("✅ Account created. You are now logged in.")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()  # 👈 Use st.rerun() instead of deprecated function
                else:
                    st.error("❌ Username already exists.")

    elif option == "Login":
        with st.sidebar.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if login_user(email, password):
                    from auth.auth_manager import get_username_by_email
                    st.session_state.logged_in = True
                    username = get_username_by_email(email)
                    st.session_state.username = username if username else email
                    st.success(f"✅ Logged in as {st.session_state.username}")
                    st.rerun()  # 👈 Refresh to show main app after login
                else:
                    st.error("❌ Invalid credentials.")
    st.stop()  # Halt the rest of the app until user is authenticated


# --- Welcome message ---
st.sidebar.success(f"Welcome, {st.session_state.username} 👋")

#----dashboard-----
def show_dashboard(username):
    st.info("Dashboard feature has been removed as per user request.")


# ------------------ Main App ------------------

questions = get_questions()
def collect_feedback(insomnia_level, song_label, user_input):
    st.markdown("## 🗣️ Share Your Feedback")
    st.write("Your input helps us improve the recommendations and understand what works best for you.")

    with st.form("feedback_form"):
        st.markdown(f"**🎵 Song Recommended:** {song_label}")
        st.markdown(f"**😴 Predicted Insomnia Level:** {insomnia_level}")

        relaxed = st.radio("Did the song help you feel relaxed?", ["Yes", "No"], index=0)
        sleep = st.radio("Did the song help you fall asleep?", ["Yes", "No"], index=0)
        sleep_quality = st.selectbox("How was your sleep quality?", ["Poor", "Average", "Good", "Excellent"])
        rating = st.slider("⭐ Rate the song experience (1 = Bad, 5 = Excellent)", 1, 5, 3)
        comments = st.text_area("Additional comments or suggestions (optional)")

        submit_feedback = st.form_submit_button("Submit Feedback ✅")

        if submit_feedback:
            feedback_entry = {
                "Username": st.session_state.username,
                "Timestamp": datetime.now().isoformat(),
                "Insomnia Level": insomnia_level,
                "Recommended Song": song_label,
                "Felt Relaxed": relaxed,
                "Fell Asleep": sleep,
                "Sleep Quality": sleep_quality,
                "Rating": rating,
                "Comments": comments,
            }

            # Add user input features to feedback entry
            for i, question in enumerate(questions):
                feedback_entry[question] = user_input[i]

            feedback_file = "data/feedback.csv"
            feedback_columns = ["Username", "Timestamp", "Insomnia Level", "Recommended Song", "Felt Relaxed", "Fell Asleep", "Sleep Quality", "Rating", "Comments"] + questions
            feedback_df = pd.DataFrame([feedback_entry], columns=feedback_columns)
            os.makedirs("data", exist_ok=True)

            if os.path.exists(feedback_file):
                feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
            else:
                feedback_df.to_csv(feedback_file, index=False)

            st.success("✅ Thank you! Your feedback has been saved.")

            with st.expander("📄 See the feedback you just submitted"):
                st.json(feedback_entry)


# def show_metrics():
#     st.subheader("📊 Model Evaluation")
#     try:
#         df = pd.read_csv("data/insomnia_synthetic.csv")
#         model, label_encoder = load_model()
#         X = df[questions]
#         y = label_encoder.transform(df["Insomnia Level"])
#         y_pred = model.predict(X)

#         acc = accuracy_score(y, y_pred)
#         st.write(f"**Accuracy:** {acc:.2f}")

#         st.text("Classification Report:")
#         st.text(classification_report(y, y_pred, target_names=label_encoder.classes_))

#         st.text("Confusion Matrix:")
#         cm = confusion_matrix(y, y_pred)
#         fig, ax = plt.subplots()
#         sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_,
#                     yticklabels=label_encoder.classes_, cmap='Blues', ax=ax)
#         st.pyplot(fig)
#     except Exception as e:
#         st.error(f"Error loading evaluation metrics: {e}")

def plot_insomnia_distribution():
    st.subheader("📈 Insomnia Level Distribution")
    try:
        df = pd.read_csv("data/insomnia_synthetic.csv")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Insomnia Level", order=["No Insomnia", "Mild", "Moderate", "Severe"], ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error loading data: {e}")

def main():
    page = st.sidebar.selectbox("Navigate", ["Home", "Dashboard"])
    
    if page == "Dashboard":
        show_dashboard(st.session_state.username)
    elif page == "Home":
        st.title("🧠 Insomnia Detection & Smart Lullaby Recommender")
        st.write("Answer the following questions (0 = None, 4 = Very Severe):")

        user_input = [st.slider(q, 0, 4, 0) for q in questions]
        num_songs = st.slider("How many songs would you like to be recommended?", 1, 10, 1)

        if st.button("Predict & Recommend"):
            model, label_encoder = load_model()
            import pandas as pd
            input_df = pd.DataFrame([user_input], columns=questions)
            prediction = model.predict(input_df)[0]
            insomnia_level = label_encoder.inverse_transform([prediction])[0]
            st.success(f"🧠 Predicted Insomnia Level: **{insomnia_level}**")

            labels, links = recommend_song_from_dataset(insomnia_level, num_songs)
            for i, (label, link) in enumerate(zip(labels, links), start=1):
                st.info(f"🎵 Recommended Song {i}: {label}")
                if link:
                    st.markdown(f"[▶️ Listen on Spotify]({link})")

            collect_feedback(insomnia_level, labels[0] if labels else "No song", user_input)

        st.markdown("---")

        from utils.helper import retrain_model_with_feedback
        try:
            train_acc, test_acc = retrain_model_with_feedback()
            st.success(f"✅ Model retrained successfully! Training accuracy: {train_acc:.2f}, Test accuracy: {test_acc:.2f}")
        except Exception as e:
            st.error(f"Error retraining model: {e}")

        # show_metrics()
        plot_insomnia_distribution()

if __name__ == "__main__":
    main()
