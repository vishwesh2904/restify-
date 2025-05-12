import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from auth.auth_manager import create_user_table, signup_user, login_user, get_username_by_email
from utils.helper import load_model, get_questions, retrain_model_with_feedback, append_to_insomnia_data
from utils.recommender import recommend_song_from_dataset
from admin.admin_panel import show_admin_panel
from utils.feedback import save_feedback

# --- Initialize DB ---
create_user_table()

# --- Session Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Authentication ---
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
                signup_result = signup_user(username, email, password)
                if signup_result:
                    st.success("✅ Account created. You are now logged in.")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("❌ Username already exists.")
    else:
        if not st.session_state.logged_in:
            with st.sidebar.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                if submit:
                    login_result = login_user(email, password)
                    if login_result:
                        st.session_state.logged_in = True
                        st.session_state.username = get_username_by_email(email) or email
                        st.success(f"✅ Logged in as {st.session_state.username}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")

# --- Logout ---
def logout_button():
    if st.sidebar.button("Logout", key="unique_logout_button"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

# --- Sidebar Welcome & Logout Button ---
if st.session_state.logged_in:
    st.sidebar.success(f"Welcome, {st.session_state.username} 👋")
    logout_button()

# --- Save user input to insomnia dataset ---
def save_insomnia_entry(user_input, insomnia_level):
    data_file = "data/insomnia_synthetic.csv"
    os.makedirs("data", exist_ok=True)

    new_data = pd.DataFrame([user_input], columns=[
        "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
        "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
        "Coping Skills", "Emotion Regulation", "Age"
    ])
    new_data["Username"] = st.session_state.username
    new_data["Timestamp"] = datetime.now().isoformat()
    new_data["Insomnia Level"] = insomnia_level

    if os.path.exists(data_file):
        existing_df = pd.read_csv(data_file)
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        combined_df = new_data

    combined_df.to_csv(data_file, index=False)

# --- Dashboard ---
def show_dashboard(username):
    st.title("📊 Insomnia Data Dashboard")
    data_path = "data/insomnia_synthetic.csv"
    if not os.path.exists(data_path):
        st.warning("Insomnia synthetic data file not found.")
        return

    df = pd.read_csv(data_path)
    st.markdown("### Insomnia Level Distribution")
    st.bar_chart(df["Insomnia Level"].value_counts())

    st.markdown("### Average Scores by Insomnia Level")
    numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    avg_scores = df.groupby("Insomnia Level")[numeric_cols].mean()
    st.dataframe(avg_scores)

    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --- Feedback Form ---
def collect_feedback(insomnia_level, song_label, user_input):
    st.markdown("## 🗣️ Share Your Feedback")
    with st.form("feedback_form"):
        st.markdown(f"**🎵 Song Recommended:** {song_label}")
        st.markdown(f"**🛌 Predicted Insomnia Level:** {insomnia_level}")
        relaxed = st.radio("Did the song help you feel relaxed?", ["Yes", "No"], index=0)
        sleep = st.radio("Did the song help you fall asleep?", ["Yes", "No"], index=0)
        sleep_quality = st.selectbox("How was your sleep quality?", ["Poor", "Average", "Good", "Excellent"], key="quality")

        rating = st.radio("⭐ Rate the song experience", [1, 2, 3, 4, 5], format_func=lambda x: "⭐" * x, key="rating")
        comments = st.text_area("Additional comments or suggestions (optional)")

        submit_feedback = st.form_submit_button("Submit Feedback ✅")
        if submit_feedback:
            feedback_entry = {
                "Username": st.session_state.username,
                "Timestamp": datetime.now().isoformat(),
                "Insomnia Level": insomnia_level,
                "Recommended Song": song_label,
                "Felt Relaxed": relaxed == "Yes",
                "Fell Asleep": sleep == "Yes",
                "Sleep Quality": sleep_quality,
                "Rating": rating,
                "Comments": comments,
            }

            for i, q in enumerate(get_questions()):
                feedback_entry[q] = user_input[i]

            feedback_file = "data/feedback.csv"
            os.makedirs("data", exist_ok=True)

            try:
                pd.DataFrame([feedback_entry]).to_csv(feedback_file, mode='a', header=not os.path.exists(feedback_file), index=False)
                st.session_state.feedback_submitted = True
                st.success("✅ Thank you! Your feedback has been saved.")
                with st.expander("📄 See your submitted feedback"):
                    st.json(feedback_entry)
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

# --- Main App ---
def main():
    if not st.session_state.logged_in:
        st.warning("Please log in to access the application features.")
        return

    user_is_admin = st.session_state.username.lower() == "admin"
    menu_options = ["Home", "Dashboard"]
    if user_is_admin:
        menu_options.append("Admin Panel")

    page = st.sidebar.selectbox("Navigate", menu_options)

    questions_with_help = {
        "Insomnia Severity": "Measures overall severity of insomnia symptoms.",
        "Sleep Quality": "Reflects how restful your sleep has been recently.",
        "Depression Level": "How often you've felt down or disinterested.",
        "Sleep Hygiene": "How well you follow good sleep practices.",
        "Negative Thoughts About Sleep": "Extent of negative thinking about sleep.",
        "Bedtime Worrying": "Worry or anxiety levels at bedtime.",
        "Stress Level": "Your general stress level recently.",
        "Coping Skills": "How effectively you deal with stressors.",
        "Emotion Regulation": "How well you manage difficult emotions.",
        "Age": "What is the age of the user?"
    }
    questions = list(questions_with_help.keys())

    if page == "Dashboard":
        show_dashboard(st.session_state.username)

    elif page == "Admin Panel":
        show_admin_panel()

    elif page == "Home":
        st.title("🧠 Insomnia Detection & Smart Lullaby Recommender")
        st.markdown("Rate each symptom from **0.0 (None)** to **4.0 (Very Severe)**:")

        user_input = [
            st.number_input(f"**{q}**\n\n_{desc}_", 0.0, 4.0, step=0.1, key=f"input_{i}", value=0.0)
            if q != "Age" else st.number_input(f"**{q}**\n\n_{desc}_", 1, 100, step=1, key=f"input_{i}", value=25)
            for i, (q, desc) in enumerate(questions_with_help.items())
        ]

        num_songs = st.number_input("Number of song recommendations:", 1, 10, 1, key="num_songs", placeholder="1 to 10")

        if st.button("Predict & Recommend"):
            try:
                model, label_encoder, scaler = load_model()

                input_df = pd.DataFrame([user_input], columns=questions)
                scaled_input = scaler.transform(input_df)
                prediction = model.predict(scaled_input)[0]
                insomnia_level = label_encoder.inverse_transform([prediction])[0]

                st.success(f"🧠 Predicted Insomnia Level: **{insomnia_level}**")

                save_insomnia_entry(user_input, insomnia_level)
                st.success("📁 Your input has been saved to the dataset.")

                labels, links, thumbnails = recommend_song_from_dataset(insomnia_level, num_songs)

                for i, (label, link, thumb) in enumerate(zip(labels, links, thumbnails), start=1):
                    st.markdown(f"### 🎵 Recommended Song {i}")
                    if thumb:
                        st.image(thumb, width=200)
                    st.markdown(f"**{label}**")
                    if link:
                        st.markdown(
                            f'<a href="{link}" target="_blank">'
                            f'<button style="background-color:#1DB954; color:white; padding:10px; border:none; border-radius:5px;">'
                            f'▶️ Listen on Spotify</button></a>',
                            unsafe_allow_html=True
                        )

                collect_feedback(insomnia_level, labels[0] if labels else "N/A", user_input)

            except Exception as e:
                st.error(f"Error during prediction or recommendation: {e}")

# --- Run App ---
main()
