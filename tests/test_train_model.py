import unittest
import pandas as pd
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_model import train_model

def train_and_save_model(df):
    # Wrapper to call train_model function for testing
    return train_model()

from utils.helper import FEATURE_COLS

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

if __name__ == "__main__":
    unittest.main()
