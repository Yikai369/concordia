"""Results saving and display functions."""

import json
import os

from projects.impression_management_standard.config import ConversationConfig
from projects.impression_management_standard.models import TurnLog


def plot_learning_dynamics(turn_logs: list[TurnLog], save_dir: str) -> None:
    """Plot learning dynamics: PE, I_t/I_hat, and learning gain.

    Creates three plots:
    1) Prediction Error across turns
    2) Score targets/estimates (I_t, I_hat) across turns
    3) Learning gain (|delta I_hat| / |PE|)

    Args:
        turn_logs: List of TurnLog entries from the simulation.
        save_dir: Directory to save plot files.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("Warning: matplotlib not available. Skipping plots.")
        print("  Install with: pip install matplotlib")
        return

    if not turn_logs:
        print("Warning: No turn data to plot.")
        return

    # Extract data
    turns = [log.turn for log in turn_logs]
    I_t = [log.audience_I for log in turn_logs]
    I_hat = [log.actor_I_hat for log in turn_logs]
    PE = [log.actor_pe for log in turn_logs]

    # Compute belief changes (delta I_hat)
    delta_I = [0.0]
    for t in range(1, len(I_hat)):
        delta_I.append(I_hat[t] - I_hat[t - 1])

    # Compute learning gain: |delta I_hat| / |PE|
    # Learning gain measures how much the belief changed relative to prediction error
    eps = 1e-6  # Small epsilon to avoid division by zero
    learning_gain = [0.0]
    for t in range(1, len(delta_I)):
        # Use previous turn's PE (PE[t-1] corresponds to turn t)
        pe_prev = abs(PE[t - 1]) if t - 1 < len(PE) else eps
        gain = abs(delta_I[t]) / (pe_prev + eps)
        learning_gain.append(gain)

    # --- Plot 1: Prediction Error ---
    # Filter out turn 1 (no PE for first turn) - filter by turn number, not array index
    pe_plot_data = [(t, abs(p)) for t, p in zip(turns, PE) if t != 1]
    if pe_plot_data:
        pe_turns, pe_values = zip(*pe_plot_data)
        plt.figure()
        plt.plot(pe_turns, pe_values, marker="o")
        plt.xlabel("Turn")
        plt.ylabel("Prediction Error: |posterior − prior|")
        plt.title("Prediction Error Across Turns")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        pe_plot_path = os.path.join(save_dir, "pe.png")
        plt.savefig(pe_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"✓ Saved plot to {pe_plot_path}")

    # --- Plot 2: I_t and I_hat ---
    plt.figure()
    plt.plot(turns, I_t, marker="x", label="True I_t")
    plt.plot(turns, I_hat, marker="o", label="Estimated I_hat")
    plt.xlabel("Turn")
    plt.ylabel("Competency Score")
    plt.legend()
    plt.title("I_t and I_hat Across Turns")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    delta_I_plot_path = os.path.join(save_dir, "delta_I.png")
    plt.savefig(delta_I_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot to {delta_I_plot_path}")

    # --- Plot 3: Learning gain ---
    plt.figure()
    plt.plot(turns, learning_gain, marker="s")
    plt.xlabel("Turn")
    plt.ylabel("Learning Gain: |delta I_hat| / |PE|")
    plt.title("Learning Gain Across Turns")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True)
    learning_gain_plot_path = os.path.join(save_dir, "learning_gain.png")
    plt.savefig(learning_gain_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved plot to {learning_gain_plot_path}")


def print_pretty_trace(turn_logs: list[TurnLog]):
    """Print a readable conversation trace similar to the example."""
    for log in turn_logs:
        print(f"[t={log.turn}] {log.speaker} → {log.listener}: {log.speaker_text}")
        if log.speaker_body:
            print(f"       Body: {log.speaker_body}")
        print(f"       {log.listener} response: {log.audience_text}")
        if log.audience_body:
            print(f"       Body: {log.audience_body}")
        print(f"       {log.listener} true I_t={log.audience_I:.2f}; {log.speaker} belief I_hat={log.actor_I_hat:.2f}, PE={log.actor_pe:+.2f}")
        if log.reflection_text:
            print(f"       {log.speaker} reflection: {log.reflection_text}")
        print()


def save_config(config: ConversationConfig, save_dir: str) -> None:
    """Save simulation configuration to JSON file.

    Args:
        config: The ConversationConfig object containing all simulation parameters.
        save_dir: Directory where the config file should be saved.
    """
    from dataclasses import asdict

    config_path = os.path.join(save_dir, "config.json")
    config_dict = asdict(config)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=2)
    print(f"✓ Saved config to {config_path}")


def save_component_logs(sim, save_dir: str) -> str | None:
    """Save Concordia component-level logs from all entities to JSON file.

    Args:
        sim: The Simulation object containing entities.
        save_dir: Directory where the log file should be saved.

    Returns:
        Path to saved file, or None if no logs were found.
    """
    component_logs = {}
    has_logs = False

    for entity in sim.entities:
        if hasattr(entity, 'get_all_logs'):
            logs = entity.get_all_logs()

            if logs:  # Only add if there are logs
                has_logs = True
                # Convert to serializable format
                serializable_logs = {}
                for channel, entries in logs.items():
                    serializable_entries = []
                    for entry in entries:
                        # Convert non-serializable objects to strings
                        if isinstance(entry, (str, int, float, bool, type(None))):
                            serializable_entries.append(entry)
                        elif isinstance(entry, (dict, list)):
                            serializable_entries.append(entry)
                        else:
                            # Convert complex objects to string representation
                            serializable_entries.append(str(entry))
                    serializable_logs[channel] = serializable_entries

                component_logs[entity.name] = {
                    'channels': serializable_logs,
                    'channel_count': len(serializable_logs),
                    'total_entries': sum(len(entries) for entries in serializable_logs.values()),
                }

    if not has_logs:
        return None

    # Save to file
    log_file = os.path.join(save_dir, 'component_logs.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(component_logs, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved component logs to {log_file}")
    return log_file


def save_results(
    config: ConversationConfig,
    turn_logs: list[TurnLog],
):
    """Save results to JSON, generate plots, and print summary."""
    from dataclasses import asdict

    # Save configuration
    save_config(config, config.save_dir)

    # Save JSON
    json_path = os.path.join(config.save_dir, config.outfile)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(log) for log in turn_logs], f, ensure_ascii=False, indent=2)
    print(f"✓ Saved log to {json_path}")

    # Generate plots if not disabled
    if not config.no_plots:
        print("\nGenerating plots...")
        plot_learning_dynamics(turn_logs, config.save_dir)

    # Print summary
    print("\n" + "="*60)
    print("Conversation Summary")
    print("="*60)
    for log in turn_logs:
        print(f"Turn {log.turn}: I_t={log.audience_I:.2f}, I_hat={log.actor_I_hat:.2f}, PE={log.actor_pe:+.2f}")
    print("="*60)

    # Print pretty trace if requested
    if config.print_trace:
        print("\n" + "="*60)
        print("Conversation Trace")
        print("="*60)
        print_pretty_trace(turn_logs)
