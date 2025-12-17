"""Test the main conversation script with mock LLM."""

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

from concordia.language_model import language_model


class MockLanguageModel(language_model.LanguageModel):
    """Mock language model for testing main script."""

    def __init__(self):
        self.call_count = 0

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: list[str] | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: float = 60.0,
        seed: int | None = None,
    ) -> str:
        """Return mock response based on prompt type."""
        self.call_count += 1

        # Evaluation prompt (audience rates actor)
        if 'rate' in prompt.lower() or 'competent' in prompt.lower():
            if self.call_count <= 2:
                return '0.7'
            else:
                return '0.8'

        # Measurement prompt (actor estimates from audience response)
        if 'estimate' in prompt.lower() and 'internal evaluation' in prompt.lower():
            if self.call_count <= 3:
                return '0.65'
            else:
                return '0.75'

        # Reflection prompt
        if 'reflection' in prompt.lower() or 'change next turn' in prompt.lower():
            return 'I will try to be more clear and demonstrate my technical skills.'

        # Act prompt (generating utterance)
        if 'DIALOGUE' in prompt or 'utterance' in prompt.lower():
            if self.call_count <= 1:
                return 'DIALOGUE: I have experience in product management and data analysis.\nBODY: Maintaining eye contact and speaking clearly'
            else:
                return 'DIALOGUE: I can demonstrate my ability to prioritize features based on user data.\nBODY: Leaning forward with confidence'

        # Response prompt (audience generates feedback)
        if 'reply' in prompt.lower() and 'reflects your evaluation' in prompt.lower():
            if '0.7' in prompt:
                return 'DIALOGUE: That sounds good. Can you tell me more about your experience?\nBODY: Nodding encouragingly'
            else:
                return 'DIALOGUE: Excellent, I\'d like to hear more about your approach.\nBODY: Showing interest'

        return 'Test response'

    def sample_choice(
        self,
        prompt: str,
        responses: list[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict]:
        """Sample a choice from available responses."""
        return (0, responses[0] if responses else '', {})


class TestMainScript(unittest.TestCase):
    """Test the main conversation script."""

    @patch('sentence_transformers.SentenceTransformer')
    @patch('pe_conversation_concordia.language_model_utils.language_model_setup')
    def test_main_script_basic(self, mock_lm_setup, mock_st):
        """Test main script with 2 turns."""
        # Setup mocks
        mock_model = MockLanguageModel()
        mock_lm_setup.return_value = mock_model

        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.1] * 384  # Mock embedding
        mock_st.return_value = mock_embedder

        # Import main script (after patching)
        import pe_conversation_concordia as main_script

        # Test with minimal arguments
        test_args = [
            '--turns', '2',
            '--llm_type', 'openai',
            '--save_dir', './test_output',
            '--no_audience_norms',  # Simplify for testing
            '--no_traits',  # Simplify for testing
            '--no_context',  # Simplify for testing
        ]

        with patch('sys.argv', ['pe_conversation_concordia.py'] + test_args):
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                try:
                    turn_logs = main_script.main()

                    # Verify results
                    self.assertIsNotNone(turn_logs)
                    self.assertGreaterEqual(len(turn_logs), 1)

                    # Check first turn log structure
                    if turn_logs:
                        log = turn_logs[0]
                        self.assertIsNotNone(log.turn)
                        self.assertIsNotNone(log.speaker)
                        self.assertIsNotNone(log.listener)
                        self.assertIsNotNone(log.speaker_text)
                        self.assertIsNotNone(log.audience_I)
                        self.assertIsNotNone(log.actor_I_hat)

                        print(f"\n✓ Main script test passed!")
                        print(f"  Generated {len(turn_logs)} turn logs")
                        print(f"  Turn 1: I_t={log.audience_I:.2f}, I_hat={log.actor_I_hat:.2f}")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.fail(f"Main script failed: {e}")


if __name__ == '__main__':
    unittest.main(verbosity=2)
