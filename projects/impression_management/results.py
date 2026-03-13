"""Results saving and display functions."""

import json
import os

from projects.impression_management.config import ConversationConfig
from projects.impression_management.models import TurnLog


def save_component_logs(entities: list[object], save_dir: str) -> str | None:
    """Save component-level logs from entities to JSON file."""
    component_logs = {}
    has_logs = False

    for entity in entities:
        if not hasattr(entity, 'get_all_logs'):
            continue

        logs = entity.get_all_logs()
        if not logs:
            continue

        has_logs = True
        serializable_logs = {}
        for channel, entries in logs.items():
            serializable_entries = []
            for entry in entries:
                if isinstance(entry, (str, int, float, bool, type(None))):
                    serializable_entries.append(entry)
                elif isinstance(entry, (dict, list)):
                    serializable_entries.append(entry)
                else:
                    serializable_entries.append(str(entry))
            serializable_logs[channel] = serializable_entries

        component_logs[entity.name] = {
            'channels': serializable_logs,
            'channel_count': len(serializable_logs),
            'total_entries': sum(len(entries) for entries in serializable_logs.values()),
        }

    if not has_logs:
        return None

    log_file = os.path.join(save_dir, 'component_logs.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(component_logs, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved component logs to {log_file}")
    return log_file


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
