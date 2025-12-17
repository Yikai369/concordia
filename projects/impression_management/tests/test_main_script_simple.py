"""Simple test for main script - just verify it can be imported and basic structure works."""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add impression_management directory to path
imp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if imp_dir not in sys.path:
    sys.path.insert(0, imp_dir)

# Mock sentence_transformers before any imports
sys.modules['sentence_transformers'] = MagicMock()

# Now we can import
from concordia.language_model import language_model


class MockLanguageModel(language_model.LanguageModel):
    """Mock language model."""

    def __init__(self):
        self.call_count = 0

    def sample_text(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        if 'rate' in prompt.lower() or 'competent' in prompt.lower():
            return '0.7' if self.call_count <= 2 else '0.8'
        if 'estimate' in prompt.lower():
            return '0.65' if self.call_count <= 3 else '0.75'
        if 'reflection' in prompt.lower():
            return 'I will try to be more clear.'
        if 'DIALOGUE' in prompt:
            return 'DIALOGUE: Test response.\nBODY: Test body language'
        if 'reply' in prompt.lower():
            return 'DIALOGUE: Good response.\nBODY: Nodding'
        return 'Test response'

    def sample_choice(self, prompt: str, responses: list[str], **kwargs) -> tuple[int, str, dict]:
        return (0, responses[0] if responses else '', {})


class TestMainScriptSimple(unittest.TestCase):
    """Simple test for main script."""

    def test_import_main_script(self):
        """Test that main script can be imported."""
        # Import utils first to make it available
        from concordia.language_model import utils as language_model_utils
        with patch.object(language_model_utils, 'language_model_setup') as mock_lm:
            mock_lm.return_value = MockLanguageModel()

            # Mock sentence_transformers
            mock_st = MagicMock()
            mock_st.return_value.encode.return_value = [0.1] * 384

            with patch('sentence_transformers.SentenceTransformer', return_value=mock_st):
                try:
                    import pe_conversation_concordia
                    print("✓ Main script imported successfully")
                    self.assertTrue(True)
                except Exception as e:
                    self.fail(f"Failed to import main script: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
