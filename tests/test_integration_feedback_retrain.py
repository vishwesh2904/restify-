import unittest
import os
import pandas as pd
from unittest.mock import patch
from utils.feedback import update_insomnia_synthetic_with_questionnaire
from utils.helper import retrain_model_with_feedback

class TestIntegrationFeedbackRetrain(unittest.TestCase):
    insomnia_csv = "data/test_insomnia_synthetic.csv"

    def setUp(self):
        # Remove test file if exists
        if os.path.exists(self.insomnia_csv):
            os.remove(self.insomnia_csv)
        # Create empty insomnia synthetic CSV with required columns
        columns = [
            "Insomnia Severity", "Sleep Quality", "Depression Level", "Sleep Hygiene",
            "Negative Thoughts About Sleep", "Bedtime Worrying", "Stress Level",
            "Coping Skills", "Emotion Regulation", "Insomnia Level"
        ]
        pd.DataFrame(columns=columns).to_csv(self.insomnia_csv, index=False)

    def tearDown(self):
        if os.path.exists(self.insomnia_csv):
            os.remove(self.insomnia_csv)

    @patch('utils.helper.retrain_model_with_feedback')
    def test_retrain_triggered_on_feedback_update(self, mock_retrain):
        mock_retrain.return_value = 0.95  # Mock accuracy

        new_feedback = {
            "Insomnia Severity": 2,
            "Sleep Quality": 3,
            "Depression Level": 1,
            "Sleep Hygiene": 3,
            "Negative Thoughts About Sleep": 1,
            "Bedtime Worrying": 1,
            "Stress Level": 2,
            "Coping Skills": 3,
            "Emotion Regulation": 3,
            "Insomnia Level": "Mild"
        }

        result = update_insomnia_synthetic_with_questionnaire(new_feedback, self.insomnia_csv)
        self.assertTrue(result)
        mock_retrain.assert_called_once()

if __name__ == "__main__":
    unittest.main()
