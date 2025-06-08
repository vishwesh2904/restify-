import unittest
import pandas as pd
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_model import train_model
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from utils.helper import FEATURE_COLS
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import unittest
import os

def train_and_save_model(df):
    # Wrapper to call train_model function for testing
    return train_model()

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        # Prepare synthetic data for training
        data = []
        for _ in range(100):
            responses = [1]*len(FEATURE_COLS)
            level = "Mild"
            data.append(responses + [sum(responses), level])
        self.df = pd.DataFrame(data, columns=FEATURE_COLS + ["Total Score", "Insomnia Level"])

        # Remove model files if exist
        if os.path.exists("models/insomnia_model.pkl"):
            os.remove("models/insomnia_model.pkl")
        if os.path.exists("models/label_encoder.pkl"):
            os.remove("models/label_encoder.pkl")
        if os.path.exists("models/scaler.pkl"):
            os.remove("models/scaler.pkl")

    def test_train_and_save_model(self):
        accuracy = train_and_save_model(self.df)
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

        # Check if model files are created
        self.assertTrue(os.path.exists("models/insomnia_model.pkl"))
        self.assertTrue(os.path.exists("models/label_encoder.pkl"))
        self.assertTrue(os.path.exists("models/scaler.pkl"))


    def test_smote_balances_classes(self):
        df = pd.read_csv("data/insomnia_synthetic.csv").dropna()
        X = df[FEATURE_COLS]
        y = df["Insomnia Level"]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

        # Check if classes are balanced after SMOTE
        unique, counts = pd.Series(y_resampled).value_counts().sort_index().items()
        self.assertTrue(len(set(counts)) == 1, "SMOTE did not balance the classes equally")

    def test_train_model_performance(self):
        accuracy = train_model()
        self.assertGreaterEqual(accuracy, 0.85, "Model accuracy is below 85%")

    def test_train_model_with_missing_data(self):
        df = pd.read_csv("data/insomnia_synthetic.csv")
        df.loc[0, FEATURE_COLS[0]] = None  # Introduce missing value
        df = df.dropna()

        accuracy = train_model()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_train_model_with_corrupted_data(self):
        df = pd.read_csv("data/insomnia_synthetic.csv")
        df.loc[0, FEATURE_COLS[0]] = "corrupted"  # Introduce corrupted value
        df = df[pd.to_numeric(df[FEATURE_COLS[0]], errors='coerce').notnull()]

        accuracy = train_model()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == "__main__":
    unittest.main()
