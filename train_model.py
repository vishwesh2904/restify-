from utils.helper import generate_data, train_and_save_model

def main():
    df = generate_data()
    train_and_save_model(df)
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    main()
