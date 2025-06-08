import pandas as pd

import os

# Define the features expected by the model
MODEL_FEATURES = [
    "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
    "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
    "Coping Skills", "Emotion Regulation"
]

# Help text for the features (can be used in UI)
questions_with_help = {
    "Insomnia Severity": "Measures the overall severity of your insomnia symptoms.",
    "Sleep Quality": "Reflects how restful and satisfying your sleep has been.",
    "Depression Level": "Assesses feelings of hopelessness or disinterest.",
    "Sleep Hygiene": "Indicates how regularly you follow good sleep habits.",
    "Negative Thoughts About Sleep": "Measures worries and doubts about sleep.",
    "Bedtime Worrying": "Assesses racing thoughts or anxiety at bedtime.",
    "Stress Level": "Indicates your general stress level recently.",
    "Coping Skills": "Measures how well you deal with stress or seek help.",
    "Emotion Regulation": "Assesses how you manage emotions under stress."
}

def standardize_columns(df, col_mapping, required_cols):
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0
    return df

def fix_feedback_csv(feedback_csv_path="data/feedback.csv", excel_path="auth/users.xlsx"):
    # Backup first
    if os.path.exists(feedback_csv_path):
        backup_path = feedback_csv_path.replace(".csv", "_backup.csv")
        shutil.copy(feedback_csv_path, backup_path)
        print(f"Backup saved at {backup_path}")

    try:
        df = pd.read_csv(feedback_csv_path)
    except FileNotFoundError:
        print(f"Feedback file {feedback_csv_path} not found.")
        return

    # Column renaming rules
    old_to_new = {
        "Difficulty falling asleep?": "Insomnia Severity",
        "Difficulty staying asleep?": "Insomnia Severity",
        "Problems waking up too early?": "Insomnia Severity",
        "Satisfaction with current sleep pattern?": "Sleep Quality",
        "Interference with daily functioning due to sleep issues?": "Stress Level",
        "Noticeability of sleep problems to others?": "Stress Level",
        "Worry/distress about current sleep issues?": "Stress Level"
    }

    # Clean and align feedback
    df = standardize_columns(df, old_to_new, MODEL_FEATURES)

    # Normalize and sanitize numeric values
    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).clip(0, 10)

    # Reorder final columns
    base_cols = ["Username", "Timestamp", "Insomnia Level", "Recommended Song", "Felt Relaxed", "Fell Asleep", "Sleep Quality", "Rating", "Comments"]
    ordered_cols = base_cols + [col for col in MODEL_FEATURES if col not in base_cols]
    df = df[[col for col in ordered_cols if col in df.columns]]

    # Save cleaned feedback
    df.to_csv(feedback_csv_path, index=False)
    print(f"Feedback CSV cleaned and saved at {feedback_csv_path}")

    # Update Excel file columns
    if os.path.exists(excel_path):
        excel_df = pd.read_excel(excel_path)
        excel_df = standardize_columns(excel_df, old_to_new, MODEL_FEATURES)
        for col in MODEL_FEATURES:
            excel_df[col] = pd.to_numeric(excel_df[col], errors='coerce').fillna(0).clip(0, 10)

        excel_cols = ["Username", "Email", "Password"] + MODEL_FEATURES
        excel_df = excel_df[[col for col in excel_cols if col in excel_df.columns]]
        excel_df.to_excel(excel_path, index=False)
        print(f"Excel file updated at {excel_path}")
    else:
        print(f"Excel file {excel_path} not found.")

def update_insomnia_synthetic_with_questionnaire(new_data: dict, insomnia_csv_path="data/insomnia_synthetic.csv"):
    # Load existing data
    try:
        df = pd.read_csv(insomnia_csv_path)
    except FileNotFoundError:
        print(f"Insomnia synthetic data file {insomnia_csv_path} not found.")
        return False

    new_df = pd.DataFrame([new_data])

    # Validate new data columns
    for col in MODEL_FEATURES + ["Insomnia Level"]:
        if col not in new_df.columns:
            print(f"Missing column in new data: {col}")
            new_df[col] = 0

    # Append and save
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(insomnia_csv_path, index=False)
    print("Insomnia synthetic data updated.")

    # Retrain model
    from utils.helper import retrain_model_with_feedback
    try:
        accuracy = retrain_model_with_feedback()
        print(f"Model retrained. Accuracy: {accuracy:.2f}")
    except Exception as e:
        print(f"Error retraining model: {e}")
        return False

    return True

def extract_valid_feedback_rows(feedback_path="data/feedback.csv"):
    try:
        df = pd.read_csv(feedback_path)
        valid = df[df["Felt Relaxed"] | df["Fell Asleep"]]
        return valid[MODEL_FEATURES + ["Insomnia Level"]]
    except Exception as e:
        print(f"Error reading feedback for training: {e}")
        return pd.DataFrame()

def update_song_recommendations(feedback_path="data/feedback.csv", song_score_path="data/song_scores.csv"):
    try:
        df = pd.read_csv(feedback_path)
        song_stats = df.groupby("Recommended Song").agg({
            "Felt Relaxed": "sum",
            "Fell Asleep": "sum",
            "Rating": "mean",
            "Username": "count"
        }).rename(columns={"Username": "Total Users"})
        song_stats["Effectiveness"] = (song_stats["Felt Relaxed"] + song_stats["Fell Asleep"]) / song_stats["Total Users"]
        song_stats = song_stats.sort_values(by=["Effectiveness", "Rating"], ascending=False)
        song_stats.to_csv(song_score_path)
        print(f"Song recommendation scores updated at {song_score_path}")
    except Exception as e:
        print(f"Failed to update song recommendation scores: {e}")

def save_feedback(new_data, file_path="data/feedback.csv"):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    else:
        df = pd.DataFrame([new_data])
    df.to_csv(file_path, index=False)        

if __name__ == "__main__":
    fix_feedback_csv()
    update_song_recommendations()
