"""Results saving and display functions."""

import json
import os

from projects.impression_management.config import ConversationConfig
from projects.impression_management.models import TurnLog


def save_results(
    config: ConversationConfig,
    turn_logs: list[TurnLog],
):
    """Save results to JSON and print summary."""
    from dataclasses import asdict

    # Save JSON
    json_path = os.path.join(config.save_dir, config.outfile)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(log) for log in turn_logs], f, ensure_ascii=False, indent=2)
    print(f"✓ Saved log to {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("Conversation Summary")
    print("="*60)
    for log in turn_logs:
        print(f"Turn {log.turn}: I_t={log.audience_I:.2f}, I_hat={log.actor_I_hat:.2f}, PE={log.actor_pe:.2f}")
    print("="*60)
