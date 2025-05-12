import unittest
import pandas as pd
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.helper import load_model, retrain_model_with_feedback
from utils.recommender import recommend_song_from_dataset

class TestModelLoadingAndPrediction(unittest.TestCase):

    def test_load_model(self):
        # Test loading model, label encoder, and scaler
        model, label_encoder, scaler = load_model()
        self.assertIsNotNone(model)
        self.assertIsNotNone(label_encoder)
        self.assertIsNotNone(scaler)

    def test_prediction_and_recommendation(self):
        # Sample input for prediction
        sample_input = [2, 3, 1, 2, 0, 1, 2, 3, 1]
        questions = [
            "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
            "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
            "Coping Skills", "Emotion Regulation"
        ]
        input_df = pd.DataFrame([sample_input], columns=questions)

        model, label_encoder, scaler = load_model()
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        insomnia_level = label_encoder.inverse_transform([prediction])[0]

        self.assertIn(insomnia_level, label_encoder.classes_)

        # Test recommendation returns proper types and doesn't exceed requested count
        labels, links, thumbnails = recommend_song_from_dataset(insomnia_level, num_songs=3)
        self.assertIsInstance(labels, list)
        self.assertIsInstance(links, list)
        self.assertIsInstance(thumbnails, list)
        self.assertLessEqual(len(labels), 3)
        self.assertLessEqual(len(links), 3)
        self.assertLessEqual(len(thumbnails), 3)

    @patch("utils.helper.SMOTE.fit_resample")
    def test_retrain_model_with_feedback(self, mock_fit_resample):
        # Mock SMOTE to avoid ValueError due to small sample size
        mock_fit_resample.return_value = (
            pd.DataFrame([[1]*9, [2]*9, [3]*9, [4]*9, [5]*9]),  # larger fake features
            pd.Series([0, 1, 0, 1, 0])          # larger fake target
        )

        accuracy = retrain_model_with_feedback()
        self.assertIsInstance(accuracy, float)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

if __name__ == "__main__":
    unittest.main()
