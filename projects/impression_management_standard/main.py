#!/usr/bin/env python3
"""
Impression Management PE Conversation - Standard Simulation Loop
----------------------------------------------------------------
Two-agent conversation system with particle filter belief tracking,
using the standard Concordia simulation loop (sim.play()).
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
from projects.impression_management_standard import config
from projects.impression_management_standard import data_extraction
from projects.impression_management_standard import results
from projects.impression_management_standard import setup
from projects.impression_management_standard import simulation_config

from concordia.prefabs.simulation import generic as simulation


def main():
    """Main entry point using standard simulation loop."""
    # Parse arguments
    cfg = config.parse_arguments()
    print(f"Output directory: {cfg.save_dir}")
    print("Using standard simulation loop (sim.play())")

    # Validate API key
    api_key = config.validate_api_key(cfg)

    # Setup components
    model = setup.setup_language_model(cfg, api_key)
    embedder, memory_bank = setup.setup_embedder_and_memory()

    # Setup random seed
    rng = random.Random(cfg.seed)

    # Create simulation config
    print("Creating simulation config...")
    sim_config = simulation_config.create_simulation_config(cfg, rng)

    # Initialize simulation
    print("Initializing simulation...")
    sim = simulation.Simulation(
        config=sim_config,
        model=model,
        embedder=embedder,
    )
    print("✓ Simulation initialized")

    # Run simulation using standard loop
    print(f"\nRunning {cfg.turns} turn conversation...")
    raw_log = []
    results_log = sim.play(
        max_steps=cfg.turns * 2,  # Each turn = actor acts, audience acts
        raw_log=raw_log,
    )
    print("✓ Simulation completed")

    # Extract and save results
    print("\nExtracting turn data...")
    turn_logs = data_extraction.extract_turn_data_from_entities(
        sim, cfg.actor_name, cfg.audience_name, cfg.turns
    )

    if not turn_logs:
        print("Warning: No turn data extracted. Check component state.")
    else:
        results.save_results(cfg, turn_logs)

    return turn_logs


if __name__ == '__main__':
    main()
