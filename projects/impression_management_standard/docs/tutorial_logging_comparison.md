# Logging Comparison: Tutorial Examples vs. Our Implementation

## Overview

This document compares how the Concordia tutorial examples handle logging versus how we've implemented logging in the `impression_management_standard` project.

---

## Tutorial Examples: How They Save Logs

### 1. Example: `pe_conversation_concordia.py`

**Location**: `examples/pe_conversation_concordia.py`

**What They Save**:
- **Turn logs** (extracted data) to JSON file
- **Pretty print** to console

**How They Do It**:
```python
# Run simulation
raw_log = []
results_log = sim.play(max_steps=args.turns * 2, raw_log=raw_log)

# Extract turn data
turn_logs = extract_turn_data_from_entities(
    sim, agent_a_name, agent_b_name, args.turns
)

# If extraction from entities failed, try from log
if not turn_logs:
    turn_logs = extract_turn_data_from_log(
        raw_log, agent_a_name, agent_b_name
    )

# Pretty print
for r in turn_logs:
    print(f'[t={r.turn}] {r.speaker} -> {r.listener}: {r.speaker_text}')
    # ... more printing

# Save JSON
with open(args.outfile, 'w', encoding='utf-8') as f:
    json.dump(
        [asdict(l) for l in turn_logs],
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f'Saved detailed log -> {args.outfile}')
```

**Key Points**:
- ✅ Saves extracted turn data (structured)
- ✅ Uses `raw_log` for extraction if entity extraction fails
- ❌ Does NOT save `raw_log` itself
- ❌ Does NOT save component logs
- ❌ Does NOT save configuration

---

### 2. Tutorial Documentation: `TUTORIAL.md`

**What It Shows**:

**Access Raw Log Data**:
```python
raw_log = []
results_log = sim.play(max_steps=10, raw_log=raw_log)

# raw_log is a list of step dictionaries
for step in raw_log:
    print(f"Step {step.get('Step', 'Unknown')}:")
    print(f"  Summary: {step.get('Summary', 'N/A')}")
    # Access entity actions, game master decisions, etc.
```

**Extract Specific Information**:
```python
from concordia.utils import helper_functions as helper_funcs

# Find specific data in the nested log structure
scores = helper_funcs.find_data_in_nested_structure(raw_log, "Player Scores")
events = helper_funcs.find_data_in_nested_structure(raw_log, "Event")
```

**Key Points**:
- ✅ Shows how to access `raw_log` (in-memory)
- ✅ Shows how to extract specific data from `raw_log`
- ❌ Does NOT show saving `raw_log` to file
- ❌ Does NOT show component log access

---

## Our Implementation: What We Save

### Files Saved by `impression_management_standard`

1. **Turn Logs** (`pe_conversation_log.json`)
   - Extracted turn data (same as tutorial examples)
   - Saved via `results.save_results()`

2. **Configuration** (`config.json`)
   - All simulation parameters
   - CLI arguments
   - Saved automatically with turn logs

3. **Plots** (`pe.png`, `delta_I.png`, `learning_gain.png`)
   - Visualization of learning dynamics
   - Optional (can disable with `--no_plots`)

4. **Information Flow History** (`information_flow_history_[timestamp].json`)
   - All LLM prompts and responses
   - Complete interaction history
   - Optional (requires `--enable_info_flow_logging`)

5. **Simplified Log** (`information_flow_simplified_[timestamp].txt`)
   - Human-readable format of information flow
   - Optional (requires `--enable_info_flow_logging --enable_simplified_log`)

6. **Component Logs** (`component_logs.json`)
   - Concordia component-level logs
   - Component state and behavior
   - Optional (requires `--save_component_logs`)

---

## Comparison Table

| Feature | Tutorial Examples | Our Implementation |
|---------|-------------------|-------------------|
| **Turn Logs (JSON)** | ✅ Yes | ✅ Yes |
| **Configuration** | ❌ No | ✅ Yes (`config.json`) |
| **Plots** | ❌ No | ✅ Yes (3 plots) |
| **Raw Log** | ⚠️ In-memory only | ⚠️ In-memory only |
| **Component Logs** | ❌ No | ✅ Yes (optional) |
| **Information Flow** | ❌ No | ✅ Yes (optional) |
| **Simplified Log** | ❌ No | ✅ Yes (optional) |

---

## What's Missing: Raw Log Saving

### Current State

**Tutorial Examples**: Don't save `raw_log` to file
**Our Implementation**: Also doesn't save `raw_log` to file

### Why Raw Log Might Be Useful

The `raw_log` contains:
- Complete step-by-step simulation trace
- Game master decisions
- Entity observations
- Action attempts
- Component state snapshots
- More detailed than extracted turn logs

### Recommendation

Consider adding `--save_raw_log` flag to save the raw log:

```python
# In main.py
if cfg.save_raw_log:
    raw_log_file = os.path.join(cfg.save_dir, 'raw_log.json')
    with open(raw_log_file, 'w', encoding='utf-8') as f:
        json.dump(raw_log, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved raw log to {raw_log_file}")
```

---

## Summary

### Tutorial Approach
- **Minimal**: Only saves extracted turn data
- **Simple**: Direct JSON dump of turn logs
- **No metadata**: Doesn't save configuration or other logs

### Our Approach
- **Comprehensive**: Saves multiple types of logs
- **Structured**: Organized with timestamps and metadata
- **Optional**: Most logs are opt-in via CLI flags
- **Reproducible**: Saves configuration for reproducibility

### Key Differences

1. **Configuration Saving**: We save `config.json` (tutorials don't)
2. **Visualization**: We generate plots (tutorials don't)
3. **Component Logs**: We can save component logs (tutorials don't)
4. **Information Flow**: We have comprehensive LLM interaction logging (tutorials don't)
5. **Raw Log**: Neither saves `raw_log` to file (could be added)

---

## Recommendations

### Already Implemented ✅
- Turn logs (same as tutorials)
- Configuration saving (better than tutorials)
- Plots (additional feature)
- Component logs (additional feature)
- Information flow history (additional feature)

### Could Be Added
- **Raw log saving**: Add `--save_raw_log` flag to save `raw_log` to JSON
- **Raw log analysis**: Add utilities to extract data from `raw_log` (like tutorial shows)

---

## Code Examples

### Tutorial Style (Minimal)
```python
# Extract and save turn logs only
turn_logs = extract_turn_data_from_entities(sim, ...)
with open('output.json', 'w') as f:
    json.dump([asdict(l) for l in turn_logs], f, indent=2)
```

### Our Style (Comprehensive)
```python
# Save multiple types of logs
results.save_results(cfg, turn_logs)  # Turn logs + config + plots
if cfg.enable_info_flow_logging:
    sim.save_information_flow_history()  # LLM interactions
if cfg.save_component_logs:
    results.save_component_logs(sim, cfg.save_dir)  # Component logs
```

---

## Conclusion

Our implementation is **more comprehensive** than the tutorial examples:

✅ **Better**: Configuration saving, plots, multiple log types
✅ **Optional**: Most features are opt-in via CLI flags
✅ **Reproducible**: Configuration saved for reproducibility
⚠️ **Missing**: Raw log saving (could be added if needed)

The tutorial examples focus on simplicity and minimal logging, while our implementation provides comprehensive logging options for debugging and analysis.
