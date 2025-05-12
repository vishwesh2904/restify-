from utils.helper import train_and_save_model

def train_model():
    import pandas as pd
    df = pd.read_csv("data/insomnia_synthetic.csv").dropna()
    accuracy = train_and_save_model(df)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    return accuracy
