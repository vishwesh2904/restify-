import pandas as pd
import os

def fix_feedback_csv(feedback_csv_path="data/feedback.csv"):
    if not os.path.exists(feedback_csv_path):
        print(f"Feedback file not found at {feedback_csv_path}")
        return

    df = pd.read_csv(feedback_csv_path)

    # Define expected columns
    questions = [
        "Difficulty falling asleep?",
        "Difficulty staying asleep?",
        "Problems waking up too early?",
        "Satisfaction with current sleep pattern?",
        "Interference with daily functioning due to sleep issues?",
        "Noticeability of sleep problems to others?",
        "Worry/distress about current sleep issues?"
    ]

    # Check if all question columns exist
    missing_cols = [q for q in questions if q not in df.columns]

    if not missing_cols:
        print("Feedback CSV already contains all required feature columns.")
        return

    # Add missing columns with default NaN values
    for col in missing_cols:
        df[col] = pd.NA

    # Save fixed CSV backup
    backup_path = feedback_csv_path + ".bak"
    os.rename(feedback_csv_path, backup_path)
    print(f"Backup of original feedback CSV saved as {backup_path}")

    # Save fixed CSV
    df.to_csv(feedback_csv_path, index=False)
    print(f"Fixed feedback CSV saved at {feedback_csv_path}")

if __name__ == "__main__":
    fix_feedback_csv()
