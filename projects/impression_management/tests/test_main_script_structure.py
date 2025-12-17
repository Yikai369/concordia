"""Test main script structure and verify it can be parsed."""

import sys
import os
import ast
import unittest

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Add impression_management directory to path
imp_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if imp_dir not in sys.path:
    sys.path.insert(0, imp_dir)


class TestMainScriptStructure(unittest.TestCase):
    """Test that main script has correct structure."""

    def test_script_exists_and_parsable(self):
        """Test that main script exists and is valid Python."""
        script_path = os.path.join(imp_dir, 'pe_conversation_concordia.py')
        self.assertTrue(os.path.exists(script_path), "Main script should exist")

        # Try to parse it as Python
        with open(script_path, 'r', encoding='utf-8') as f:
            try:
                ast.parse(f.read())
                print("✓ Main script is valid Python")
            except SyntaxError as e:
                self.fail(f"Main script has syntax errors: {e}")

    def test_script_has_main_function(self):
        """Test that main script has a main() function."""
        script_path = os.path.join(imp_dir, 'pe_conversation_concordia.py')
        with open(script_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())

        # Check for main function
        has_main = any(
            isinstance(node, ast.FunctionDef) and node.name == 'main'
            for node in ast.walk(tree)
        )
        self.assertTrue(has_main, "Main script should have a main() function")
        print("✓ Main script has main() function")

    def test_script_has_turnlog_dataclass(self):
        """Test that main script has TurnLog dataclass."""
        script_path = os.path.join(imp_dir, 'pe_conversation_concordia.py')
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('class TurnLog', content, "Should have TurnLog dataclass")
        self.assertIn('@dataclass', content, "Should use @dataclass decorator")
        print("✓ Main script has TurnLog dataclass")

    def test_script_has_cli_args(self):
        """Test that main script has CLI argument parsing."""
        script_path = os.path.join(imp_dir, 'pe_conversation_concordia.py')
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('argparse', content, "Should use argparse")
        self.assertIn('--turns', content, "Should have --turns argument")
        self.assertIn('--model', content, "Should have --model argument")
        print("✓ Main script has CLI argument parsing")


if __name__ == '__main__':
    unittest.main(verbosity=2)
