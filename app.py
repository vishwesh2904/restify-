import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from auth.auth_manager import create_user_table, signup_user, login_user, get_username_by_email
import utils.helper as helper
from utils.helper import get_questions, retrain_model_with_feedback, append_to_insomnia_data, load_model
from utils.recommender import recommend_song_from_dataset
from admin.admin_panel import show_admin_panel
from utils.feedback import save_feedback

# --- Initialize DB ---
create_user_table()

# --- Set Page Config for Insomnia Theme ---
st.set_page_config(page_title="Restify | Insomnia Recommender", page_icon="🌙", layout="centered")

import streamlit as st

# Custom CSS for better visibility and aesthetics
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;  /* Darker background for modern look */
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 16px;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        padding-left: 4rem;
        padding-right: 4rem;
        max-width: 900px;
        margin: auto;
        background: linear-gradient(135deg, #1f2937, #111827);
        border-radius: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.6);
    }

    h1, h2, h3 {
        color: #facc15 !important;  /* Bright amber for headings */
        font-weight: 700;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.7);
    }

    .stNumberInput label, .stMarkdown p {
        font-size: 18px !important;
        color: #cbd5e1 !important;
        font-weight: 600;
    }

    input[type="number"] {
        background-color: #374151 !important;
        color: #f9fafb !important;
        border-radius: 6px;
        border: 1px solid #4b5563;
        padding: 6px 8px;
        transition: border-color 0.3s ease;
    }
    input[type="number"]:focus {
        border-color: #fbbf24 !important;
        outline: none;
        box-shadow: 0 0 8px #fbbf24;
    }

    button, .stButton>button {
        background-color: #fbbf24 !important;
        color: #1f2937 !important;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 12px rgba(251, 191, 36, 0.5);
        transition: background-color 0.3s ease, color 0.3s ease;
        cursor: pointer;
    }
    button:hover, .stButton>button:hover {
        background-color: #f59e0b !important;
        color: #111827 !important;
    }

    /* Fix feedback form question label visibility */
    .stRadio > label, .stRadio > div > label, .stSelectbox > label, .stTextArea > label {
        color: #e0e0e0 !important;
        font-weight: 600;
        font-size: 16px;
    }

    /* Force all text inside feedback form to be visible */
    form#feedback_form * {
        color: #e0e0e0 !important;
    }

    /* Make radio button option text (Yes/No) white */
    form#feedback_form .stRadio > div > label > div[data-testid="stMarkdownContainer"] > p {
        color: #ffffff !important;
    }

    footer {
        text-align: center;
        color: #888;
        margin-top: 30px;
        font-size: 14px;
    }
    </style>
""", unsafe_allow_html=True)


# --- Session Initialization ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Authentication ---
if not st.session_state.logged_in:
    st.sidebar.title("🔐 Authentication")
    option = st.sidebar.radio("Login/Signup", ("Login", "Signup"), key="auth_radio")

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

if st.session_state.logged_in:
    st.sidebar.success(f"Welcome, {st.session_state.username} 👋")
    logout_button()

# Remaining code stays unchanged...

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
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    mask = None
    if corr.shape[0] == corr.shape[1]:
        mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",
        center=0,
        mask=mask,
        annot_kws={"size": 10},
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight='bold')
    st.pyplot(fig)

# --- Feedback Form ---
def collect_feedback(insomnia_level, song_label, user_input):
    st.markdown("## 🗣️ Share Your Feedback")
    with st.form("feedback_form"):
        st.markdown(f"<div style='font-size:20px; font-weight:700; color:#facc15; margin-bottom:10px;'>🎵 Song Recommended: <span style='color:#e0e0e0;'>{song_label}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:20px; font-weight:700; color:#facc15; margin-bottom:20px;'>🛌 Predicted Insomnia Level: <span style='color:#e0e0e0;'>{insomnia_level}</span></div>", unsafe_allow_html=True)

        container_style = "background: #1f2937; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 700px; margin-bottom: 20px; color: #e0e0e0;"
        with st.container():
            st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
            relaxed = st.radio("Did the song help you feel relaxed?", ["Yes", "No"], index=0, horizontal=True)
            sleep = st.radio("Did the song help you fall asleep?", ["Yes", "No"], index=0, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
            st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)
            sleep_quality = st.selectbox("How was your sleep quality?", ["Poor", "Average", "Good", "Excellent"])
            rating = st.radio("⭐ Rate the song experience", [1, 2, 3, 4, 5], format_func=lambda x: "⭐" * x, horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        comments_style = "background: #1f2937; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); max-width: 700px; margin-bottom: 20px; color: #e0e0e0;"
        comments = st.text_area("Additional comments or suggestions (optional)", height=100, placeholder="Your comments here...", key="feedback_comments")
        st.markdown(f"<div style='{comments_style}'></div>", unsafe_allow_html=True)

        submit_style = "background-color: #fbbf24; color: #1f2937; font-weight: 700; border-radius: 8px; padding: 12px 24px; border: none; box-shadow: 0 4px 12px rgba(251, 191, 36, 0.5); cursor: pointer; transition: background-color 0.3s ease;"
        submit_feedback = st.form_submit_button("Submit Feedback ✅", help="Click to submit your feedback")
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
    menu_options = ["Home", "Dashboard", "Feedback"]
    if user_is_admin:
        menu_options.append("Admin Panel")

    page = st.sidebar.selectbox("Navigate", menu_options, index=menu_options.index(st.session_state.get("page", "Home")))
    st.session_state.page = page

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

    elif page == "Feedback":
        insomnia_level = st.session_state.get("feedback_insomnia_level")
        song_label = st.session_state.get("feedback_song_label")
        user_input = st.session_state.get("feedback_user_input")

        if "feedback_submitted" in st.session_state:
            del st.session_state["feedback_submitted"]

        if insomnia_level and song_label and user_input:
            collect_feedback(insomnia_level, song_label, user_input)
        else:
            st.info("Please get a song recommendation first from the Home page.")

    elif page == "Home":
        st.markdown("""
    <div style='
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.6rem;
        color: #2c3e50;
        font-weight: 600;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    '>
        💤 Questions to Assess Insomnia Symptoms and Severity
    </div>
""", unsafe_allow_html=True)

        # st.markdown("### Insomnia Symptoms Assessment")
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

        st.markdown(
    """
    <div style="
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 8px 12px; border-radius: 10px; color: white; font-weight: 600;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 15px; box-shadow: 1px 1px 8px rgba(0,0,0,0.12);
        max-width: 600px; margin-bottom: 10px;">
        <div style="display: flex; align-items: center; justify-content: center;">
            <span style="margin: 0 10px 0 0;">Rate symptoms:</span>
            <div style="display: flex; gap: 18px;">
                <div style="text-align:center;">
                    <div style="font-weight:700;">0</div>
                    <div style="font-size:12px;">None</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">1</div>
                    <div style="font-size:12px;">Very Mild</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">2</div>
                    <div style="font-size:12px;">Mild</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">3</div>
                    <div style="font-size:12px;">Moderate</div>
                </div>
                <div style="font-weight:700;">|</div>
                <div style="text-align:center;">
                    <div style="font-weight:700;">4</div>
                    <div style="font-size:12px;">Severe</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


        user_input = [
            st.number_input(f"**{q}**\n\n_{desc}_", 0.0, 4.0, step=0.1, key=f"input_{i}", value=0.0)
            if q != "Age" else st.number_input(f"**{q}**\n\n_{desc}_", 1, 100, step=1, key=f"input_{i}", value=25)
            for i, (q, desc) in enumerate(questions_with_help.items())
        ]

        num_songs = st.number_input("Number of song recommendations:", 1, 10, 1, key="num_songs")

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

                # Save feedback context
                st.session_state.feedback_insomnia_level = insomnia_level
                st.session_state.feedback_song_label = labels[0] if labels else "N/A"
                st.session_state.feedback_user_input = user_input

            except Exception as e:
                st.error(f"Error during prediction or recommendation: {e}")

        # Show Go to Feedback button only if prediction is made
        if (
            "feedback_insomnia_level" in st.session_state and
            "feedback_song_label" in st.session_state and
            "feedback_user_input" in st.session_state
        ):
            st.markdown("---")
            if st.button("🗣️ Go to Feedback Form", key="go_to_feedback_bottom"):
                st.session_state.page = "Feedback"
                st.rerun()


# --- Run App ---
main()
