# Simplified Information Flow Log - Implementation Plan

**Date**: 2025-12-27
**Purpose**: Create a simplified, human-readable log format alongside the existing detailed JSON log

---

## Overview

Create a simplified log format that shows:
- **Receiver**: Agent name receiving the information
- **Input text**: The prompt sent to the model
- **Output**: The model's response

Format: `[receiver: XX, input: XXX, output: XXX]`

---

## Requirements

1. ✅ **Keep Original Log**: The detailed JSON log must remain unchanged
2. ✅ **Create Simplified Log**: Generate a new simplified format
3. ✅ **Human-Readable**: Easy to read and understand at a glance
4. ✅ **Optional**: Can be enabled/disabled via configuration
5. ✅ **Separate File**: Save as a separate file (e.g., `.txt` or `.md`)

---

## Proposed Format

### Format 1: Simple Text Format

```
=== Information Flow Log ===
Simulation: [simulation_name]
Date: [timestamp]
Total Interactions: [count]

---

[Turn 1] [receiver: Jane, input: "You are Jane. You want to evaluate...", output: "0.8"]

[Turn 1] [receiver: John, input: "PERSONALITY TRAITS:\n- Detail-focused...", output: "DIALOGUE: I have successfully prioritized features...\nBODY: I maintain a confident posture..."]

[Turn 2] [receiver: Jane, input: "You are Jane. You want to evaluate...", output: "0.6"]

...
```

### Format 2: Markdown Format (More Structured)

```markdown
# Information Flow Log

**Simulation**: [simulation_name]
**Date**: [timestamp]
**Total Interactions**: [count]

## Turn 1

### Interaction 1
- **Receiver**: Jane
- **Input**:
  ```
  You are Jane. You want to evaluate...
  ```
- **Output**:
  ```
  0.8
  ```

### Interaction 2
- **Receiver**: John
- **Input**:
  ```
  PERSONALITY TRAITS:
  - Detail-focused...
  ```
- **Output**:
  ```
  DIALOGUE: I have successfully prioritized features...
  BODY: I maintain a confident posture...
  ```

## Turn 2
...
```

### Format 3: Compact Line Format (Recommended)

```
=== Information Flow Log ===
Date: 2025-12-27 23:04:32
Total Interactions: 71

[Turn 0] [receiver: Jane] [input: "You are Jane. You are in an alternative world..."] [output: "Understood. What topic would you like to discuss..."]
[Turn 0] [receiver: Jane] [input: "You are Jane. You want to evaluate. Evaluate the interviewee's competence..."] [output: "The goal is to evaluate the interviewee's competence...\n\nMy evaluation is: 0.2."]
[Turn 0] [receiver: John] [input: "You are John. Be perceived as competent..."] [output: "0.6"]
[Turn 1] [receiver: Jane] [input: "You are Jane. You want to evaluate..."] [output: "0.8"]
...
```

**Advantages**:
- One line per interaction
- Easy to grep/search
- Compact but readable
- Can be easily parsed if needed

---

## Implementation Plan

### Step 1: Add Method to InformationFlowHistoryBank

**File**: `concordia/utils/information_flow_history.py`

**New Method**: `generate_simplified_log(format: str = 'compact') -> str`

**Functionality**:
- Iterate through all interactions
- Format each interaction according to chosen format
- Group by turn (optional)
- Return formatted string

**Code Structure**:
```python
def generate_simplified_log(
    self,
    format: str = 'compact',  # 'compact', 'markdown', 'text'
    max_input_length: int = 200,  # Truncate long inputs
    max_output_length: int = 200,  # Truncate long outputs
    group_by_turn: bool = True,
) -> str:
    """Generate a simplified, human-readable log.

    Args:
        format: Output format ('compact', 'markdown', 'text')
        max_input_length: Maximum characters for input (0 = no limit)
        max_output_length: Maximum characters for output (0 = no limit)
        group_by_turn: Whether to group interactions by turn

    Returns:
        Formatted log string
    """
```

### Step 2: Add Truncation Helper

**Function**: `_truncate_text(text: str, max_length: int) -> str`

- Truncate long prompts/responses
- Add `...` indicator if truncated
- Preserve newlines in a readable way

### Step 3: Add Save Method

**New Method**: `save_simplified_log(filepath: str | None = None, format: str = 'compact') -> str`

**Functionality**:
- Generate simplified log
- Save to file (`.txt` for compact/text, `.md` for markdown)
- Return filepath

### Step 4: Integrate into Simulation

**File**: `concordia/prefabs/simulation/generic.py`

**Changes**:
- Add `enable_simplified_log: bool = False` parameter
- Add `simplified_log_format: str = 'compact'` parameter
- Call `save_simplified_log()` after simulation completes (if enabled)

### Step 5: Add CLI Option

**File**: `projects/impression_management_standard/config.py`

**Changes**:
- Add `--enable_simplified_log` flag
- Add `--simplified_log_format` option (choices: 'compact', 'markdown', 'text')
- Pass to `ConversationConfig`

### Step 6: Update Models

**File**: `projects/impression_management_standard/models.py`

**Changes**:
- Add `enable_simplified_log: bool = False` to `ConversationConfig`
- Add `simplified_log_format: str = 'compact'` to `ConversationConfig`

---

## Format Details

### Compact Format (Recommended)

**Structure**:
```
[Turn N] [receiver: AgentName] [input: "prompt text..."] [output: "response text..."]
```

**Example**:
```
[Turn 0] [receiver: Jane] [input: "You are Jane. You want to evaluate. Evaluate the interviewee's competence (0=not competent, 1=fully competent).."] [output: "0.2"]
[Turn 0] [receiver: John] [input: "PERSONALITY TRAITS:\n- Detail-focused (1/3): I tend to focus..."] [output: "DIALOGUE: I have successfully prioritized features by analyzing user data...\nBODY: I maintain a confident posture..."]
```

**Features**:
- One line per interaction
- Easy to read
- Easy to grep: `grep "receiver: Jane" simplified_log.txt`
- Can be parsed with simple regex

### Markdown Format

**Structure**:
```markdown
## Turn N

### Interaction M
- **Receiver**: AgentName
- **Component**: ComponentName (if available)
- **Phase**: phase_name (if available)
- **Input**:
  ```
  prompt text...
  ```
- **Output**:
  ```
  response text...
  ```
```

**Features**:
- More structured
- Better for documentation
- Can include metadata (component, phase)
- Good for GitHub/readable reports

### Text Format

**Structure**:
```
Turn N:
  Interaction M:
    Receiver: AgentName
    Input: prompt text...
    Output: response text...
```

**Features**:
- Simple indentation
- Easy to read
- Less structured than markdown

---

## Configuration Options

### Options to Add

1. **Enable/Disable**: `--enable_simplified_log`
2. **Format**: `--simplified_log_format {compact,markdown,text}`
3. **Truncation**: `--simplified_log_max_length 200` (optional, default: no limit)
4. **Group by Turn**: Always group by turn (default: true)

### Default Behavior

- **Disabled by default** (to avoid clutter)
- **Format**: `compact` (most readable)
- **No truncation** (show full text)
- **Group by turn** (easier to follow)

---

## File Naming

### Pattern

- **Compact/Text**: `information_flow_simplified_[timestamp].txt`
- **Markdown**: `information_flow_simplified_[timestamp].md`

### Location

- Same directory as detailed JSON log
- Saved in `save_dir` (e.g., `temp/2025-12-27_23-04-32/`)

---

## Implementation Details

### Truncation Logic

```python
def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text while preserving readability."""
    if max_length == 0 or len(text) <= max_length:
        return text

    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.8:  # If space is reasonably close
        truncated = truncated[:last_space]

    return truncated + "..."
```

### Formatting Logic

```python
def _format_compact(interaction: ModelInteraction, max_input: int, max_output: int) -> str:
    """Format interaction in compact format."""
    receiver = interaction.agent_name
    input_text = _truncate_text(interaction.prompt, max_input)
    output_text = _truncate_text(interaction.response, max_output)

    # Escape brackets in text
    input_text = input_text.replace('[', '\\[').replace(']', '\\]')
    output_text = output_text.replace('[', '\\[').replace(']', '\\]')

    turn_str = f"Turn {interaction.turn}" if interaction.turn is not None else "Turn ?"
    return f"[{turn_str}] [receiver: {receiver}] [input: \"{input_text}\"] [output: \"{output_text}\"]"
```

### Grouping Logic

```python
def _group_by_turn(interactions: list[ModelInteraction]) -> dict[int, list[ModelInteraction]]:
    """Group interactions by turn number."""
    grouped = {}
    for interaction in interactions:
        turn = interaction.turn if interaction.turn is not None else -1
        if turn not in grouped:
            grouped[turn] = []
        grouped[turn].append(interaction)
    return grouped
```

---

## Example Output

### Compact Format Example

```
=== Information Flow Log ===
Date: 2025-12-27 23:04:32
Total Interactions: 71
Agents: Jane (13), IMPE Conversation Rules (48), John (10)

[Turn 0] [receiver: Jane] [input: "You are Jane. You are in an alternative world in the year 3025 where there is a new set of cultural norms..."] [output: "Understood. What topic would you like to discuss or what purpose are we aiming to achieve in this interaction?"]
[Turn 0] [receiver: Jane] [input: "You are Jane. You want to evaluate. Evaluate the interviewee's competence (0=not competent, 1=fully competent).."] [output: "The goal is to evaluate the interviewee's competence for the Product Manager role. Based on the provided utterance, the interviewee focuses on interpretations and awareness of dynamics rather than demonstrating competencies related to the role's criteria. There is no evidence of technical understanding, prioritization skills, communication ability, or data-driven decision-making.\n\nMy evaluation is: 0.2."]
[Turn 0] [receiver: John] [input: "You are John. Be perceived as competent by the interviewer (0=not competent, 1=fully competent). Aim for 1.0.."] [output: "0.6"]
[Turn 1] [receiver: Jane] [input: "You are Jane. You want to evaluate..."] [output: "0.8"]
[Turn 1] [receiver: John] [input: "PERSONALITY TRAITS:\n- Detail-focused (1/3): I tend to focus..."] [output: "DIALOGUE: I have successfully prioritized features by analyzing user data, which informed product decisions that aligned with our strategic vision.\nBODY: I maintain a confident posture with a slight nod to emphasize understanding."]
...
```

---

## Benefits

1. ✅ **Quick Overview**: See all interactions at a glance
2. ✅ **Easy Search**: `grep "receiver: Jane"` to find all Jane's interactions
3. ✅ **Readable**: Much easier than parsing JSON
4. ✅ **Optional**: Doesn't clutter if not needed
5. ✅ **Flexible**: Multiple format options

---

## Testing Checklist

- [ ] Generate simplified log in compact format
- [ ] Generate simplified log in markdown format
- [ ] Generate simplified log in text format
- [ ] Verify truncation works correctly
- [ ] Verify grouping by turn works
- [ ] Verify file is saved in correct location
- [ ] Verify original JSON log is unchanged
- [ ] Test with `--enable_simplified_log` flag
- [ ] Test with `--simplified_log_format` option
- [ ] Verify log is readable and well-formatted

---

## Files to Modify

1. `concordia/utils/information_flow_history.py`
   - Add `generate_simplified_log()` method
   - Add `save_simplified_log()` method
   - Add helper functions for formatting

2. `concordia/prefabs/simulation/generic.py`
   - Add `enable_simplified_log` parameter
   - Add `simplified_log_format` parameter
   - Call `save_simplified_log()` after simulation

3. `projects/impression_management_standard/config.py`
   - Add `--enable_simplified_log` argument
   - Add `--simplified_log_format` argument

4. `projects/impression_management_standard/models.py`
   - Add fields to `ConversationConfig`

5. `projects/impression_management_standard/main.py`
   - Pass config to simulation
   - Handle simplified log saving

---

## Future Enhancements (Optional)

1. **Filtering**: Filter by agent, component, phase, turn
2. **Statistics**: Add summary statistics (total tokens, average response length, etc.)
3. **Color Coding**: Add colors for different agents/phases (if terminal output)
4. **Interactive Viewer**: Create a simple viewer script to browse logs
5. **Comparison**: Compare logs from different runs

---

## Recommendation

**Recommended Format**: **Compact** format
- Most readable
- Easy to search
- One line per interaction
- Can be easily parsed if needed

**Implementation Priority**:
1. ✅ Compact format (highest priority)
2. ⚠️ Markdown format (nice to have)
3. ⚠️ Text format (nice to have)
