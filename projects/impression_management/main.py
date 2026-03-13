#!/usr/bin/env python3
"""
Impression Management PE Conversation in Concordia Framework
------------------------------------------------------------
Two-agent conversation system with particle filter belief tracking,
cultural norms, personality traits, and interview context.
"""

import os
import sys


class _StreamTee:
  """Mirror writes to multiple text streams."""

  def __init__(self, *streams):
    self._streams = streams

  def write(self, data):
    for stream in self._streams:
      stream.write(data)
    return len(data)

  def flush(self):
    for stream in self._streams:
      stream.flush()

  def isatty(self):
    for stream in self._streams:
      if hasattr(stream, 'isatty') and stream.isatty():
        return True
    return False

  def __getattr__(self, name):
    # Delegate standard stream attributes/methods to the first stream.
    return getattr(self._streams[0], name)

# Try to load .env file if python-dotenv is available
try:
  from dotenv import load_dotenv

  # Load .env file from project directory
  env_path = os.path.join(os.path.dirname(__file__), '.env')
  if os.path.exists(env_path):
    try:
      load_dotenv(env_path, override=False)
    except Exception:
      # Silently ignore .env parsing errors (e.g., empty file, malformed)
      pass
except ImportError:
  # python-dotenv not installed, skip .env loading
  pass

# Import project modules
# Handle imports whether running from project root or script directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
  sys.path.insert(0, _project_root)

# Now import project modules
from projects.impression_management import config
from projects.impression_management import conversation
from projects.impression_management import data_extraction
from projects.impression_management import entities
from projects.impression_management import results
from projects.impression_management import setup


def main():
 """Main entry point."""
 # Parse arguments
 cfg = config.parse_arguments()
 trail_path = os.path.join(cfg.save_dir, 'print_trail.txt')
 original_stdout = sys.stdout
 original_stderr = sys.stderr

 with open(trail_path, 'w', encoding='utf-8') as trail_file:
  sys.stdout = _StreamTee(original_stdout, trail_file)
  sys.stderr = _StreamTee(original_stderr, trail_file)
  try:
   print(f'Output directory: {cfg.save_dir}')
   print(f'Print trail file: {trail_path}')

   # Validate API key
   api_key = config.validate_api_key(cfg)

   # Setup components
   model = setup.setup_language_model(cfg, api_key)
   embedder, memory_bank = setup.setup_embedder_and_memory()

   # Create goals
   goal_actor, goal_audience = entities.create_goals(cfg)

   # Prepare traits and norms
   cultural_norms, actor_has_traits, audience_has_traits = (
     entities.prepare_traits_and_norms(cfg)
   )

   # Create entities
   actor, audience = entities.create_entities(
     cfg,
     goal_actor,
     goal_audience,
     cultural_norms,
     actor_has_traits,
     audience_has_traits,
     model,
     memory_bank,
   )

   # Create game master (not used in manual execution, but kept for future use)
   game_master = entities.create_game_master(cfg, actor, audience, model, memory_bank)

   # Run conversation
   turn_summaries = conversation.run_conversation(cfg, actor, audience)

   # Extract and save results
   print('\nExtracting turn data...')
   turn_logs = data_extraction.extract_turn_data_from_entities(
     actor,
     audience,
     cfg.turns,
     turn_summaries=turn_summaries,
   )

   if cfg.save_component_logs:
    results.save_component_logs([actor, audience], cfg.save_dir)

   results.save_results(cfg, turn_logs)

   return turn_logs
  finally:
   sys.stdout = original_stdout
   sys.stderr = original_stderr


if __name__ == '__main__':
  main()
