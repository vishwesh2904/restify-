from utils.helper import retrain_model_with_feedback
from scripts.generate_realistic_synthetic_data import generate_realistic_insomnia_synthetic_data

def train_model():
    # Generate a larger, more realistic synthetic dataset
    generate_realistic_insomnia_synthetic_data(n_samples=5000)

    # Retrain model with feedback, hyperparameter tuning, and SMOTE
    accuracy = retrain_model_with_feedback()

    print("="*40)
    print("Model Evaluation Metrics after improved training:")
    print("="*40)
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    print("="*40)
    return accuracy
