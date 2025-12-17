#!/usr/bin/env python3
"""
Impression Management PE Conversation in Concordia Framework
------------------------------------------------------------
Two-agent conversation system with particle filter belief tracking,
cultural norms, personality traits, and interview context.
"""

import os
import random
import sys

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project directory
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        try:
            load_dotenv(env_path, override=False)
        except Exception as e:
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
    print(f"Output directory: {cfg.save_dir}")

    # Validate API key
    api_key = config.validate_api_key(cfg)

    # Setup components
    model = setup.setup_language_model(cfg, api_key)
    embedder, memory_bank = setup.setup_embedder_and_memory()

    # Setup random seed
    rng = random.Random(cfg.seed)

    # Create goals
    goal_actor, goal_audience = entities.create_goals(cfg)

    # Prepare traits and norms
    trait_scores_actor, trait_scores_audience, cultural_norms, traits = (
        entities.prepare_traits_and_norms(cfg, rng)
    )

    # Create entities
    actor, audience = entities.create_entities(
        cfg,
        goal_actor,
        goal_audience,
        trait_scores_actor,
        trait_scores_audience,
        cultural_norms,
        traits,
        model,
        memory_bank,
    )

    # Create game master (not used in manual execution, but kept for future use)
    game_master = entities.create_game_master(cfg, actor, audience, model, memory_bank)

    # Run conversation
    conversation.run_conversation(cfg, actor, audience)

    # Extract and save results
    print("\nExtracting turn data...")
    turn_logs = data_extraction.extract_turn_data_from_entities(actor, audience, cfg.turns)
    results.save_results(cfg, turn_logs)

    return turn_logs


if __name__ == '__main__':
    main()
