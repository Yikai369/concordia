# Feature Comparison: Example vs. Current Implementation

## Overview

This document compares the simple PE conversation example (`examples/pe_conversation_openai.py`) with the current standard implementation (`impression_management_standard`) to identify missing features.

## Core Features Comparison

### ✅ Features Present in Both

1. **PE (Prediction Error) Calculation**: Both compute prediction error
2. **Reflection Generation**: Both generate reflections based on PE
3. **Turn-based Logging**: Both log per-turn data
4. **JSON Output**: Both save results to JSON
5. **Goal-based Behavior**: Both use goals to guide agent behavior

### ⚠️ Features Missing in Current Implementation

#### 1. **Pretty Print Trace** (Important for Study)

**Example:**
```python
# Pretty print concise trace
for r in runlog:
    print(f"[t={r.turn}] {r.speaker} → {r.listener}: {r.speaker_text}")
    print(f"       {r.listener} observed estimate={r.listener_estimate:.2f}, PE={r.listener_pe:+.2f}")
    print(f"       {r.listener} reflection: {r.listener_reflection}\n")
```

**Current Implementation:**
```python
# Only prints summary table
for log in turn_logs:
    print(f"Turn {log.turn}: I_t={log.audience_I:.2f}, I_hat={log.actor_I_hat:.2f}, PE={log.actor_pe:.2f}")
```

**Impact**: The example provides a readable conversation trace that shows the flow of the conversation, making it easier to understand what happened. The current implementation only shows a summary table.

**Recommendation**: Add a pretty print trace function similar to the example.

#### 2. **Signed PE Display** (Important for Analysis)

**Example:**
- Shows signed PE: `PE={r.listener_pe:+.2f}` (can be positive or negative)
- Format: `+0.15` or `-0.08`

**Current Implementation:**
- Stores absolute PE: `actor_pe = abs(pe_rec.pe)`
- Always positive, loses direction information

**Impact**: Signed PE is more informative because it shows:
- **Positive PE**: Belief is underestimating (I_hat < I_t) → need to improve more
- **Negative PE**: Belief is overestimating (I_hat > I_t) → need to be more realistic

**Recommendation**: Store signed PE in the log and display it with sign in the trace.

#### 3. **Full Conversation Flow Display**

**Example:**
- Shows speaker → listener with arrow
- Shows full utterance text
- Shows listener's estimate, PE, and reflection
- Indented formatting for readability

**Current Implementation:**
- Only shows summary metrics (I_t, I_hat, PE)
- Doesn't show the actual conversation text in the trace
- Doesn't show reflection text in the trace

**Impact**: The example makes it easy to read the conversation flow and understand what each agent said and how they reacted. The current implementation requires opening the JSON file to see the actual conversation.

**Recommendation**: Add a pretty print function that shows the full conversation flow.

## Enhanced Features in Current Implementation

### ✅ Features Present Only in Current Implementation

1. **Body Language**: Current implementation tracks and logs body language descriptions
2. **Particle Filter**: More sophisticated belief tracking than simple PE
3. **Cultural Norms**: Influence agent behavior
4. **Personality Traits**: Influence agent behavior
5. **Effective Sample Size (ESS)**: Particle filter quality metric
6. **True Hidden State (I_t)**: Audience's actual evaluation (not just estimate)

## Recommendations

### High Priority

1. **Add Pretty Print Trace**: Implement a function similar to the example that shows:
   - Full conversation flow with arrows
   - Speaker utterances
   - Listener estimates and PE (with sign)
   - Reflections

2. **Store Signed PE**: Change `actor_pe` to store signed PE instead of absolute value, or add a separate `actor_pe_signed` field.

### Medium Priority

3. **Add Conversation Text to Summary**: Include actual utterance text in the summary output, not just metrics.

4. **Add Reflection to Summary**: Show reflection text in the summary output.

## Implementation Plan

### Step 1: Update TurnLog Model

Add signed PE field (or change existing to signed):
```python
@dataclass
class TurnLog:
    # ... existing fields ...
    actor_pe: float  # Change from absolute to signed
    # OR add:
    actor_pe_signed: float  # Signed PE (can be negative)
```

### Step 2: Update Data Extraction

Store signed PE instead of absolute:
```python
# Current:
actor_pe = abs(pe_rec.pe) if pe_rec else 0.0

# Change to:
actor_pe = pe_rec.pe if pe_rec else 0.0  # Keep sign
```

### Step 3: Add Pretty Print Function

Add to `results.py`:
```python
def print_pretty_trace(turn_logs: list[TurnLog]):
    """Print a readable conversation trace."""
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
```

### Step 4: Update save_results()

Call pretty print after summary:
```python
def save_results(config, turn_logs):
    # ... existing save code ...

    # Print summary
    print("\n" + "="*60)
    print("Conversation Summary")
    print("="*60)
    for log in turn_logs:
        print(f"Turn {log.turn}: I_t={log.audience_I:.2f}, I_hat={log.actor_I_hat:.2f}, PE={log.actor_pe:+.2f}")
    print("="*60)

    # Add pretty print trace
    print("\n" + "="*60)
    print("Conversation Trace")
    print("="*60)
    print_pretty_trace(turn_logs)
```

## Conclusion

The current implementation has more sophisticated features (particle filter, cultural norms, body language) but is missing the **pretty print trace** feature that makes the example easy to read and understand. Adding this feature would improve the usability of the standard implementation for studies and analysis.

