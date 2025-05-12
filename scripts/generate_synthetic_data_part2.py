# Moderate Insomnia samples - tighter range, less overlap with Mild and Severe
    for _ in range(samples_per_category):
        insomnia_severity = random.choice([2,3])
        sleep_quality = max(0, 3 - insomnia_severity + random.choice([0,1]))
        depression_level = min(3, insomnia_severity + random.choice([0,1]))
        sleep_hygiene = max(0, 3 - random.randint(0, insomnia_severity))
        negative_thoughts = min(3, insomnia_severity + random.choice([0,1]))
        bedtime_worrying = min(3, insomnia_severity + random.choice([0,1]))
        stress_level = min(3, insomnia_severity + random.choice([0,1]))
        coping_skills = max(0, 3 - insomnia_severity + random.choice([0,1]))
        emotion_regulation = max(0, 3 - insomnia_severity + random.choice([0,1]))

        responses = [
            insomnia_severity,
            sleep_quality,
            depression_level,
            sleep_hygiene,
            negative_thoughts,
            bedtime_worrying,
            stress_level,
            coping_skills,
            emotion_regulation
        ]
        total = sum(responses)
        level = "Moderate"
        data.append(responses + [total, level])

    # Severe Insomnia samples - higher values, less noise
    for _ in range(samples_per_category):
        insomnia_severity = random.choice([3,4])
        sleep_quality = max(0, 2 - insomnia_severity + random.choice([0]))
        depression_level = min(4, insomnia_severity + random.choice([0,1]))
        sleep_hygiene = max(0, 2 - random.randint(0, insomnia_severity))
        negative_thoughts = min(4, insomnia_severity + random.choice([0,1]))
        bedtime_worrying = min(4, insomnia_severity + random.choice([0,1]))
        stress_level = min(4, insomnia_severity + random.choice([0]))
        coping_skills = max(0, 2 - insomnia_severity + random.choice([0]))
        emotion_regulation = max(0, 2 - insomnia_severity + random.choice([0]))

        responses = [
            insomnia_severity,
            sleep_quality,
            depression_level,
            sleep_hygiene,
            negative_thoughts,
            bedtime_worrying,
            stress_level,
            coping_skills,
            emotion_regulation
        ]
        total = sum(responses)
        level = "Severe"
        data.append(responses + [total, level])

    df = pd.DataFrame(data, columns=QUESTIONS + ["Total Score", "Insomnia Level"])
    os.makedirs("data", exist_ok=True)
    # Overwrite existing data with new high-quality data
    df.to_csv("data/insomnia_synthetic.csv", index=False)
    print(f"High-quality synthetic insomnia data with {n_samples} entries generated and saved to data/insomnia_synthetic.csv")

def generate_feedback_data(n_samples=350):
    usernames = [f"user{i+1001}" for i in range(n_samples)]  # New usernames to avoid duplicates
    base_date = datetime(2025, 6, 1)
    data = []
    for i in range(n_samples):
        username = usernames[i]
        timestamp = base_date + timedelta(days=i)
        insomnia_severity = random.choices([0,1,2,3,4], weights=[0.3,0.3,0.2,0.15,0.05])[0]
        if insomnia_severity <= 1:
            insomnia_level = random.choice(["No Insomnia", "Mild"])
            sleep_quality = random.choice(["Good", "Excellent"])
            felt_relaxed = True
            fell_asleep = True
            rating = random.randint(4,5)
            comments = random.choice(["Very soothing", "Fell asleep quickly", "Relaxing", "Very calming", "Helpful", "Fell asleep fast"])
        elif insomnia_severity == 2 or insomnia_severity == 3:
            insomnia_level = "Moderate"
            sleep_quality = random.choice(["Average", "Good"])
            felt_relaxed = random.choice([True, False])
            fell_asleep = random.choice([True, False])
            rating = random.randint(2,4)
            comments = random.choice(["Helped a bit", "Relaxing", "Helpful", "Very soothing", "Fell asleep quickly"])
        else:
            insomnia_level = "Severe"
            sleep_quality = random.choice(["Poor", "Average"])
            felt_relaxed = False
            fell_asleep = False
            rating = random.randint(1,3)
            comments = random.choice(["Not effective", "Did not help", "Did not help at all", "No improvement"])

        recommended_song = f"Lullaby by Artist {random.choice(['A','B','C','D','E','F','G','H','I','J'])}"
        depression_level = min(4, insomnia_severity + random.choice([-1,0,1]))
        sleep_hygiene = max(0, 4 - random.randint(0, insomnia_severity))
        negative_thoughts = min(4, insomnia_severity + random.choice([0,1]))
        bedtime_worrying = min(4, insomnia_severity + random.choice([0,1]))
        stress_level = min(4, insomnia_severity + random.choice([-1,0,1]))
        coping_skills = max(0, 4 - insomnia_severity + random.choice([-1,0,1]))
        emotion_regulation = max(0, 4 - insomnia_severity + random.choice([-1,0,1]))

        row = [
            username,
            timestamp.isoformat(),
            insomnia_level,
            recommended_song,
            felt_relaxed,
            fell_asleep,
            sleep_quality,
            rating,
            comments,
            insomnia_severity,
            sleep_quality,  # Note: This duplicates the string sleep quality, matching original CSV structure
            depression_level,
            sleep_hygiene,
            negative_thoughts,
            bedtime_worrying,
            stress_level,
            coping_skills,
            emotion_regulation
        ]
        data.append(row)
    columns = [
        "Username", "Timestamp", "Insomnia Level", "Recommended Song", "Felt Relaxed", "Fell Asleep",
        "Sleep Quality", "Rating", "Comments", "Insomnia Severity", "Sleep Quality", "Depression Level",
        "Sleep Hygiene", "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
        "Coping Skills", "Emotion Regulation"
    ]
    df = pd.DataFrame(data, columns=columns)
    os.makedirs("data", exist_ok=True)
    # Append to existing data
    if os.path.exists("data/feedback.csv"):
        existing_df = pd.read_csv("data/feedback.csv")
        existing_df = existing_df.loc[:,~existing_df.columns.duplicated()]
        existing_df = existing_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        # Reset index of existing_df to unique values to avoid index conflicts
        existing_df.index = pd.Index(range(len(existing_df)))
        df.index = pd.Index(range(len(df)))
        # Remove duplicate columns from df before concatenation
        df = df.loc[:,~df.columns.duplicated()]
        df = pd.concat([existing_df, df], ignore_index=True)
    df.to_csv("data/feedback.csv", index=False)
    print(f"Feedback data with {n_samples} new entries generated and appended to data/feedback.csv")

if __name__ == "__main__":
    generate_insomnia_synthetic_data()
    generate_feedback_data()
