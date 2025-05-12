import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# Constants for questions
QUESTIONS = [
    "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
    "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
    "Coping Skills", "Emotion Regulation", "Age"
]

INSOMNIA_LEVELS = ["No Insomnia", "Mild", "Moderate", "Severe"]

def generate_realistic_insomnia_synthetic_data(n_samples=1000):
    data = []
    samples_per_category = n_samples // 4  # Equal samples for each category

    # Age distribution: normal distribution centered at 40, std dev 12, clipped between 18 and 65
    def generate_age():
        return int(np.clip(np.random.normal(40, 12), 18, 65))

    # Helper function to generate correlated features based on insomnia severity level
    def generate_features(insomnia_level):
        # Base means and std devs for each feature by insomnia level
        # Values scaled 0-4
        base_means = {
            "No Insomnia": [0, 4, 0, 4, 0, 0, 0, 4, 4],
            "Mild": [1.5, 3, 1, 3, 1, 1, 1, 3, 3],
            "Moderate": [2.5, 2, 2, 2, 2, 2, 2, 2, 2],
            "Severe": [3.5, 1, 3, 1, 3, 3, 3, 1, 1]
        }
        base_stds = {
            "No Insomnia": [0.5]*9,
            "Mild": [0.7]*9,
            "Moderate": [0.8]*9,
            "Severe": [0.5]*9
        }

        means = base_means[insomnia_level]
        stds = base_stds[insomnia_level]

        features = []
        for mean, std in zip(means, stds):
            val = np.random.normal(mean, std)
            val = np.clip(val, 0, 4)
            features.append(val)

        # Add some correlation: e.g. higher insomnia severity correlates with higher depression, stress, negative thoughts
        # Adjust depression level, stress level, negative thoughts about sleep based on insomnia severity
        idx_depression = 2
        idx_negative_thoughts = 4
        idx_stress = 6

        severity_index = INSOMNIA_LEVELS.index(insomnia_level)
        features[idx_depression] = np.clip(features[idx_depression] + 0.3 * severity_index, 0, 4)
        features[idx_negative_thoughts] = np.clip(features[idx_negative_thoughts] + 0.3 * severity_index, 0, 4)
        features[idx_stress] = np.clip(features[idx_stress] + 0.3 * severity_index, 0, 4)

        # Coping skills and emotion regulation inversely correlated with insomnia severity
        idx_coping = 7
        idx_emotion = 8
        features[idx_coping] = np.clip(features[idx_coping] - 0.3 * severity_index, 0, 4)
        features[idx_emotion] = np.clip(features[idx_emotion] - 0.3 * severity_index, 0, 4)

        return features

    for insomnia_level in INSOMNIA_LEVELS:
        for _ in range(samples_per_category):
            features = generate_features(insomnia_level)
            age = generate_age()

            # Insert age as last feature
            features.append(age)

            total_score = sum(features[:-1])  # exclude age from total score
            data.append(features + [total_score, insomnia_level])

    df = pd.DataFrame(data, columns=QUESTIONS + ["Total Score", "Insomnia Level"])
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/insomnia_synthetic.csv", index=False)
    print(f"Realistic synthetic insomnia data with {n_samples} entries generated and saved to data/insomnia_synthetic.csv")

if __name__ == "__main__":
    generate_realistic_insomnia_synthetic_data()
