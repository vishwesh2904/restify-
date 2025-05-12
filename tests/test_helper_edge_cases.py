import unittest
import os
import pandas as pd
from utils import helper

class TestHelperEdgeCases(unittest.TestCase):

    def setUp(self):
        # Ensure clean data file
        if os.path.exists("data/insomnia_synthetic.csv"):
            os.remove("data/insomnia_synthetic.csv")

    def test_append_to_insomnia_data_with_empty_entry(self):
        # Append empty dict should create file with no rows
        helper.append_to_insomnia_data({})
        self.assertTrue(os.path.exists("data/insomnia_synthetic.csv"))
        df = pd.read_csv("data/insomnia_synthetic.csv")
        self.assertEqual(len(df), 1)  # One row with NaNs

    def test_generate_data_with_zero_samples(self):
        df = helper.generate_data(n_samples=0)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), 0)

    def test_load_model_files_missing(self):
        # Temporarily rename model files if exist
        renamed = []
        for fname in ["model/insomnia_model.pkl", "model/label_encoder.pkl", "model/scaler.pkl"]:
            if os.path.exists(fname):
                os.rename(fname, fname + ".bak")
                renamed.append(fname)
        try:
            with self.assertRaises(FileNotFoundError):
                helper.load_model()
        finally:
            # Restore files
            for fname in renamed:
                os.rename(fname + ".bak", fname)

    def test_retrain_model_with_feedback_empty_file(self):
        # Create empty insomnia_synthetic.csv
        pd.DataFrame().to_csv("data/insomnia_synthetic.csv", index=False)
        with self.assertRaises(ValueError):
            helper.retrain_model_with_feedback()

if __name__ == "__main__":
    unittest.main()
