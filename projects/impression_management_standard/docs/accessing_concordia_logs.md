# Accessing Original Concordia Component Logs

## Overview

The Concordia framework provides a **component-level logging system** that allows components to publish log data to channels. This is separate from the **Information Flow History** system (which logs all LLM interactions). This document explains how to access and use the original Concordia logging system.

---

## What is Component-Level Logging?

The original Concordia logging system:

- **Component-Based**: Individual components can implement `ComponentWithLogging` interface
- **Channel-Based**: Components publish data to named channels
- **In-Memory**: Logs are stored in a `Measurements` object during simulation
- **Optional**: Components choose what to log (not all components log)
- **Latest State**: Focuses on the latest state of each component

### Key Concepts

1. **`EntityAgentWithLogging`**: Agent class that collects logs from components
2. **`ComponentWithLogging`**: Interface that components can implement to enable logging
3. **`Measurements`**: In-memory registry that stores log channels
4. **Channels**: Named data streams where components publish log entries

---

## How It Works

### 1. Component Logging Setup

Components that implement `ComponentWithLogging` can register a logging channel:

```python
class MyComponent(ComponentWithLogging):
    def set_logging_channel(self, channel):
        """Called by EntityAgentWithLogging to register logging channel."""
        self._logging_channel = channel

    def some_method(self):
        # Component publishes data to its channel
        if hasattr(self, '_logging_channel'):
            self._logging_channel("Some log data")
```

### 2. Agent Log Collection

`EntityAgentWithLogging` automatically:
- Creates a `Measurements` object
- Registers logging channels for components that implement `ComponentWithLogging`
- Provides methods to retrieve logs

### 3. Channel Names

- Context components: Use their component name as channel name
- Act component: Uses `'__act__'` as channel name
- Context processor: Uses `'__context_processor__'` as channel name

---

## Accessing Logs in Your Code

### Method 1: Access Logs from Simulation Entities

After running a simulation, you can access logs from entities:

```python
from projects.impression_management_standard import config, setup, simulation_config
from concordia.prefabs.simulation import generic as simulation
import random

# Setup simulation (same as main.py)
cfg = config.parse_arguments()
api_key = config.validate_api_key(cfg)
model = setup.setup_language_model(cfg, api_key)
embedder, memory_bank = setup.setup_embedder_and_memory()
rng = random.Random(cfg.seed)
sim_config = simulation_config.create_simulation_config(cfg, rng)

# Create simulation
sim = simulation.Simulation(
    config=sim_config,
    model=model,
    embedder=embedder,
)

# Run simulation
sim.play(max_steps=cfg.turns * 2)

# Access logs from entities
for entity in sim.entities:
    if hasattr(entity, 'get_all_logs'):
        # Get all logs (all channels, all entries)
        all_logs = entity.get_all_logs()
        print(f"\n=== All logs for {entity.name} ===")
        for channel_name, log_entries in all_logs.items():
            print(f"\nChannel: {channel_name}")
            print(f"  Entries: {len(log_entries)}")
            if log_entries:
                print(f"  Last entry: {log_entries[-1]}")

    if hasattr(entity, 'get_last_log'):
        # Get latest log (all channels, last entry only)
        last_log = entity.get_last_log()
        print(f"\n=== Latest log for {entity.name} ===")
        for channel_name, last_entry in last_log.items():
            print(f"  {channel_name}: {last_entry}")
```

### Method 2: Access Logs During Simulation

You can also access logs during the simulation by modifying the main script:

```python
# In main.py, after sim.play()
results_log = sim.play(
    max_steps=cfg.turns * 2,
    raw_log=raw_log,
)

# Access component logs
print("\n=== Component Logs ===")
for entity in sim.entities:
    if hasattr(entity, 'get_all_logs'):
        all_logs = entity.get_all_logs()
        print(f"\n{entity.name} component logs:")
        for channel, entries in all_logs.items():
            print(f"  {channel}: {len(entries)} entries")
```

### Method 3: Save Logs to File

**✅ Now Available**: Component log saving is now integrated into `main.py`. Use the `--save_component_logs` flag to enable it.

You can save component logs to a JSON file using the CLI flag:

**Using CLI Flag (Recommended)**:
```bash
python projects/impression_management_standard/main.py --turns 3 --save_component_logs
```

This will automatically save component logs to `component_logs.json` in the output directory.

**Programmatic Access** (if you want to customize):
```python
from projects.impression_management_standard import results

# After simulation
component_log_file = results.save_component_logs(sim, cfg.save_dir)
if component_log_file:
    print(f"Component logs saved to {component_log_file}")
else:
    print("No component logs found")
```

---

## Example: Complete Log Access Script

**Note**: Component log saving is now integrated into `main.py` via the `--save_component_logs` flag. The example below shows how to access logs programmatically if you need custom processing.

Here's a complete example that accesses and saves component logs:

```python
#!/usr/bin/env python3
"""Example: Accessing Concordia component logs."""

import os
import json
import random
import sys

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from projects.impression_management_standard import config, setup, simulation_config
from concordia.prefabs.simulation import generic as simulation


def save_component_logs(sim, save_dir: str):
    """Save component logs from all entities to JSON file."""
    component_logs = {}

    for entity in sim.entities:
        if hasattr(entity, 'get_all_logs'):
            logs = entity.get_all_logs()

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

    # Save to file
    log_file = os.path.join(save_dir, 'component_logs.json')
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(component_logs, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved component logs to {log_file}")
    return log_file


def print_log_summary(sim):
    """Print summary of component logs."""
    print("\n=== Component Log Summary ===")

    for entity in sim.entities:
        if hasattr(entity, 'get_all_logs'):
            all_logs = entity.get_all_logs()
            print(f"\n{entity.name}:")
            print(f"  Channels: {len(all_logs)}")

            for channel_name, entries in all_logs.items():
                print(f"    - {channel_name}: {len(entries)} entries")
                if entries:
                    # Show last entry (truncated if too long)
                    last_entry = entries[-1]
                    entry_str = str(last_entry)
                    if len(entry_str) > 100:
                        entry_str = entry_str[:100] + "..."
                    print(f"      Last: {entry_str}")

        if hasattr(entity, 'get_last_log'):
            last_log = entity.get_last_log()
            if last_log:
                print(f"  Latest log channels: {list(last_log.keys())}")


def main():
    """Main function to run simulation and access logs."""
    # Parse arguments
    cfg = config.parse_arguments()
    print(f"Output directory: {cfg.save_dir}")

    # Validate API key
    api_key = config.validate_api_key(cfg)

    # Setup components
    model = setup.setup_language_model(cfg, api_key)
    embedder, memory_bank = setup.setup_embedder_and_memory()
    rng = random.Random(cfg.seed)

    # Create simulation
    sim_config = simulation_config.create_simulation_config(cfg, rng)
    sim = simulation.Simulation(
        config=sim_config,
        model=model,
        embedder=embedder,
    )

    # Run simulation
    print(f"\nRunning {cfg.turns} turn conversation...")
    raw_log = []
    sim.play(max_steps=cfg.turns * 2, raw_log=raw_log)
    print("✓ Simulation completed")

    # Access and print logs
    print_log_summary(sim)

    # Save logs to file
    save_component_logs(sim, cfg.save_dir)

    return sim


if __name__ == '__main__':
    main()
```

---

## Understanding Log Structure

### `get_all_logs()` Return Value

Returns a dictionary mapping channel names to lists of log entries:

```python
{
    'channel_name_1': [entry1, entry2, entry3, ...],
    'channel_name_2': [entry1, entry2, ...],
    '__act__': [entry1, entry2, ...],
    ...
}
```

### `get_last_log()` Return Value

Returns a dictionary mapping channel names to the last log entry:

```python
{
    'channel_name_1': last_entry,
    'channel_name_2': last_entry,
    '__act__': last_entry,
    ...
}
```

### Log Entry Format

Log entries are whatever the component publishes. They can be:
- Strings
- Dictionaries
- Complex objects (may need conversion for JSON serialization)

---

## Which Components Log?

Not all components implement logging. Components that do log include:

- **Act Components**: May log action attempts
- **Context Components**: May log context they provide
- **Memory Components**: May log memory operations
- **Custom Components**: If they implement `ComponentWithLogging`

**Note**: The Impression Management PE components may or may not implement logging. Check the component source code to see if they log.

---

## Comparison: Component Logs vs. Information Flow History

| Feature | Component Logs (Original) | Information Flow History (New) |
|---------|---------------------------|--------------------------------|
| **Scope** | Component-level state | Model-level interactions |
| **What's Logged** | Component-specific data | All LLM prompts/responses |
| **Persistence** | ❌ In-memory only | ✅ Saved to JSON files |
| **Automatic** | ⚠️ Component-dependent | ✅ Automatic (all calls) |
| **Comprehensiveness** | ⚠️ Varies by component | ✅ Complete (all interactions) |
| **Access Method** | `entity.get_all_logs()` | `sim.get_information_flow_history()` |
| **Use Case** | Component state tracking | LLM interaction debugging |

### When to Use Each

**Use Component Logs when:**
- You want to track component-specific state
- You're debugging component behavior
- You need the latest state of components
- Components explicitly log useful information

**Use Information Flow History when:**
- You want to see all LLM prompts and responses
- You're debugging model interactions
- You need complete interaction history
- You want persistent logs for analysis

---

## Limitations

### 1. Not All Components Log

Many components don't implement `ComponentWithLogging`, so they won't have logs.

### 2. In-Memory Only

Logs are lost after the simulation ends unless you save them manually.

### 3. Component-Dependent

What gets logged depends on what each component chooses to log. There's no guarantee of completeness.

### 4. No Standard Format

Log entries can be any format the component chooses (string, dict, object, etc.).

### 5. No Turn Tracking

Component logs don't automatically track which turn they're from (unlike Information Flow History).

---

## Tips

1. **Check if Component Logs**: Before accessing logs, check if the entity has logging:
   ```python
   if hasattr(entity, 'get_all_logs'):
       logs = entity.get_all_logs()
   ```

2. **Save Early**: If you want to preserve logs, save them before the simulation object is destroyed.

3. **Convert for JSON**: Complex objects in logs may need conversion before saving to JSON.

4. **Use Both Systems**: Component logs and Information Flow History complement each other:
   - Component logs: Component state and behavior
   - Information Flow History: Complete LLM interaction trace

5. **Check Component Source**: To see what a component logs, check its source code for `ComponentWithLogging` implementation.

---

## Example Output

When you run the example script, you might see:

```
=== Component Log Summary ===

John:
  Channels: 2
    - __act__: 5 entries
      Last: "I have successfully prioritized features..."
    - __memory__: 10 entries
      Last: {"conversation": [...], "evaluations": [...]}
  Latest log channels: ['__act__', '__memory__']

Jane:
  Channels: 1
    - __act__: 5 entries
      Last: "The interviewee displayed moderate competence..."
  Latest log channels: ['__act__']
```

---

## References

- **EntityAgentWithLogging**: `concordia/agents/entity_agent_with_logging.py`
- **ComponentWithLogging**: `concordia/typing/entity_component.py`
- **Measurements**: `concordia/utils/measurements.py`
- **Information Flow History**: See `docs/improvements_todo.md` (Information Flow History Bank section)

---

## Summary

The original Concordia component logging system provides:
- ✅ In-memory component state tracking
- ✅ Channel-based log organization
- ✅ Latest state access via `get_last_log()`
- ✅ Full history access via `get_all_logs()`

**To access logs:**
1. Get entities from simulation: `sim.entities`
2. Check if entity has logging: `hasattr(entity, 'get_all_logs')`
3. Retrieve logs: `entity.get_all_logs()` or `entity.get_last_log()`
4. Save to file if needed (manual conversion may be required)

**Remember**: Component logs are complementary to Information Flow History - use both for complete visibility into your simulation!
