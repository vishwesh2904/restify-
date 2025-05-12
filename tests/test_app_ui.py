import unittest
from unittest.mock import patch, MagicMock
import app

class TestAppUI(unittest.TestCase):

    @patch("app.st")
    def test_home_page_inputs_and_buttons(self, mock_st):
        # Setup session state
        mock_st.session_state = MagicMock()
        mock_st.session_state.logged_in = True
        mock_st.session_state.username = "testuser"

        # Mock number inputs for symptoms and age
        mock_st.number_input = MagicMock(side_effect=[0.0]*9 + [25, 3])
        mock_st.button = MagicMock(return_value=True)
        mock_st.success = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.image = MagicMock()
        mock_st.error = MagicMock()
        mock_st.expander = MagicMock(return_value=MagicMock())
        mock_st.form_submit_button = MagicMock(return_value=False)

        with patch("app.load_model") as mock_load_model, \
             patch("app.recommend_song_from_dataset") as mock_recommend, \
             patch("app.save_insomnia_entry") as mock_save_entry, \
             patch("app.collect_feedback") as mock_collect_feedback:

            mock_load_model.return_value = (MagicMock(), MagicMock(), MagicMock())
            mock_recommend.return_value = (["Song1"], ["link1"], ["thumb1"])

            app.main()

            mock_load_model.assert_called_once()
            mock_save_entry.assert_called_once()
            mock_recommend.assert_called_once()
            mock_collect_feedback.assert_called_once()
            mock_st.success.assert_any_call("🧠 Predicted Insomnia Level: **No Insomnia**")
            mock_st.success.assert_any_call("📁 Your input has been saved to the dataset.")

if __name__ == "__main__":
    unittest.main()
