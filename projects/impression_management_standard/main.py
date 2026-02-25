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
from projects.impression_management_standard import constants
from projects.impression_management_standard import data_extraction
from projects.impression_management_standard import results
from projects.impression_management_standard import setup
from projects.impression_management_standard import simulation_config
from projects.impression_management_standard import utils

from concordia.prefabs.simulation import generic as simulation


def _run_question_checks(model, turn_logs, actor_name: str, audience_name: str):
    """Run optional question checks: situation and personality summary per agent (2 LLM calls per agent)."""
    if not turn_logs:
        return None
    # Build a short conversation summary for context
    lines = []
    for log in turn_logs:
        lines.append(f"Turn {log.turn}: {log.speaker} said: {log.speaker_text[:200]}")
        lines.append(f"  {log.listener} responded: {log.audience_text[:200]}")
    conv_summary = "\n".join(lines)

    def _ask(prompt: str) -> str:
        try:
            out = model.sample_text(prompt)
            return (out or "").strip()
        except Exception as e:
            print(f"Warning: Question check LLM call failed: {e}")
            return ""

    # Actor: situation and personality
    actor_situation_prompt = f"""Based on this conversation summary, in one or two sentences: what kind of situation is {actor_name} in?

Conversation:
{conv_summary}

Answer briefly:"""
    actor_personality_prompt = f"""Based on this conversation summary, in one or two sentences: what kind of person does {actor_name} come across as?

Conversation:
{conv_summary}

Answer briefly:"""
    # Audience: situation and personality
    audience_situation_prompt = f"""Based on this conversation summary, in one or two sentences: what kind of situation is {audience_name} in?

Conversation:
{conv_summary}

Answer briefly:"""
    audience_personality_prompt = f"""Based on this conversation summary, in one or two sentences: what kind of person does {audience_name} come across as?

Conversation:
{conv_summary}

Answer briefly:"""

    return {
        "actor_context_summary": _ask(actor_situation_prompt),
        "actor_personality_summary": _ask(actor_personality_prompt),
        "audience_context_summary": _ask(audience_situation_prompt),
        "audience_personality_summary": _ask(audience_personality_prompt),
    }


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

    # Resolve traits: from file, or constants, or None if disabled
    if cfg.no_traits:
        traits = None
    elif cfg.traits_file:
        traits = utils.load_traits_from_spreadsheet(cfg.traits_file)
        print(f"Loaded {len(traits)} traits from {cfg.traits_file}")
    else:
        traits = constants.ALL_TRAITS

    # Create simulation config
    print("Creating simulation config...")
    sim_config = simulation_config.create_simulation_config(cfg, rng, traits_override=traits)

    # Initialize simulation
    print("Initializing simulation...")
    sim = simulation.Simulation(
        config=sim_config,
        model=model,
        embedder=embedder,
        enable_information_flow_logging=cfg.enable_info_flow_logging,
        information_flow_save_dir=cfg.save_dir if cfg.enable_info_flow_logging else None,
    )
    if cfg.enable_info_flow_logging:
        print("[OK] Information flow logging enabled")
    print("[OK] Simulation initialized")

    # Run simulation using standard loop
    print(f"\nRunning {cfg.turns} turn conversation...")
    raw_log = []
    results_log = sim.play(
        max_steps=cfg.turns * 2,  # Each turn = actor acts, audience acts
        raw_log=raw_log,
    )
    print("[OK] Simulation completed")

    # Extract and save results
    print("\nExtracting turn data...")
    turn_logs, actor_traits, audience_traits = data_extraction.extract_turn_data_from_entities(
        sim, cfg.actor_name, cfg.audience_name, cfg.turns,
        use_trait_paragraph=cfg.use_trait_paragraph,
    )

    # Optional question checks (once per run, 2 LLM calls per agent)
    question_checks = None
    if cfg.enable_question_checks and turn_logs:
        print("\nRunning question checks (situation + personality per agent)...")
        question_checks = _run_question_checks(
            model, turn_logs, cfg.actor_name, cfg.audience_name
        )
        if question_checks:
            print("[OK] Question checks completed")

    if not turn_logs:
        print("Warning: No turn data extracted. Check component state.")
    else:
        results.save_results(
            cfg, turn_logs,
            actor_traits=actor_traits,
            audience_traits=audience_traits,
            question_checks=question_checks,
        )

    # Save information flow history if enabled
    if cfg.enable_info_flow_logging:
        print("\nSaving information flow history...")
        history_file = sim.save_information_flow_history()
        if history_file:
            print(f"[OK] Saved information flow history to {history_file}")

        # Save simplified log if enabled
        if cfg.enable_simplified_log:
            print("\nGenerating simplified log...")
            history_bank = sim.get_information_flow_history_bank()
            if history_bank:
                try:
                    simplified_file = history_bank.save_simplified_log(
                        format=cfg.simplified_log_format,
                    )
                    print(f"[OK] Saved simplified log to {simplified_file}")
                except Exception as e:
                    print(f"Warning: Failed to save simplified log: {e}")
            else:
                print("Warning: Information flow history bank not available for simplified log")

    # Save component logs if enabled
    if cfg.save_component_logs:
        print("\nSaving component logs...")
        try:
            component_log_file = results.save_component_logs(sim, cfg.save_dir)
            if component_log_file is None:
                print("Note: No component logs found (components may not implement ComponentWithLogging)")
        except Exception as e:
            print(f"Warning: Failed to save component logs: {e}")

    return turn_logs


if __name__ == '__main__':
    main()
