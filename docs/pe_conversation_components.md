# PE Conversation Components Documentation

## Overview

The PE (Prediction Error) conversation system implements a goal-driven adaptive conversation mechanism where agents:
1. **Estimate** the current state based on partner's responses
2. **Calculate** prediction error (PE = ideal - estimate)
3. **Reflect** on how to reduce PE
4. **Act** by generating utterances that minimize PE

This document describes the component structure and methods in `concordia/components/agent/pe_conversation.py`.

---

## Data Structures

### `Goal`
Represents the agent's goal for the conversation.

**Fields:**
- `name: str` - Name of the goal (e.g., "likability")
- `description: str` - Detailed description of the goal
- `ideal: float` - Ideal value on the goal dimension (default: 1.0)

**Example:**
```python
goal = Goal(
    name="likability",
    description="Be perceived as likable by the partner (0=not liked, 1=fully liked). Aim for 1.0.",
    ideal=1.0
)
```

### `Utterance`
Represents a single conversation utterance.

**Fields:**
- `turn: int` - Turn number when the utterance occurred
- `speaker: str` - Name of the speaker
- `text: str` - The utterance text

### `PERecord`
Stores a prediction error calculation record.

**Fields:**
- `turn: int` - Turn number
- `partner_text: str` - The partner's text that triggered this estimation
- `estimate: float` - Estimated current state (0.0 to 1.0)
- `pe: float` - Prediction error (ideal - estimate)

### `ReflectionRecord`
Stores a reflection generated to reduce PE.

**Fields:**
- `turn: int` - Turn number
- `text: str` - Reflection text

---

## Component Constants

- `DEFAULT_PE_MEMORY_COMPONENT_KEY = 'PE_Memory'`
- `DEFAULT_PE_ESTIMATION_COMPONENT_KEY = 'PE_Estimation'`
- `DEFAULT_PE_REFLECTION_COMPONENT_KEY = 'PE_Reflection'`

---

## Component Classes

### 1. `PEMemoryComponent`

**Purpose:** Central storage component for PE conversation data.

**Inheritance:**
- `action_spec_ignored.ActionSpecIgnored`
- `entity_component.ComponentWithLogging`

**Initialization:**
```python
def __init__(
    self,
    goal: Goal,
    recent_k: int = 3,
    pre_act_label: str = 'PE Memory',
)
```

**Parameters:**
- `goal: Goal` - The agent's goal
- `recent_k: int` - Number of recent items to retrieve (default: 3)
- `pre_act_label: str` - Label for pre_act output (default: 'PE Memory')

**Internal State:**
- `_goal: Goal` - Agent's goal
- `_recent_k: int` - Window size for recent history
- `_conversation: list[Utterance]` - Conversation history
- `_pe_history: list[PERecord]` - PE calculation history
- `_reflections: list[ReflectionRecord]` - Reflection history

#### Methods

##### `add_utterance(turn: int, speaker: str, text: str) -> None`
Adds a conversation utterance to memory.

**Parameters:**
- `turn: int` - Turn number
- `speaker: str` - Speaker name
- `text: str` - Utterance text

**Usage:**
```python
memory.add_utterance(turn=1, speaker="Agent A", text="Hello!")
```

##### `add_pe_record(turn: int, partner_text: str, estimate: float, pe: float) -> None`
Adds a PE record to history.

**Parameters:**
- `turn: int` - Turn number
- `partner_text: str` - Partner's text that triggered estimation
- `estimate: float` - Estimated state (0.0-1.0)
- `pe: float` - Prediction error

**Usage:**
```python
memory.add_pe_record(turn=1, partner_text="Hi there!", estimate=0.8, pe=0.2)
```

##### `add_reflection(turn: int, text: str) -> None`
Adds a reflection record.

**Parameters:**
- `turn: int` - Turn number
- `text: str` - Reflection text

##### `get_recent_conversation(k: int | None = None) -> list[Utterance]`
Retrieves the most recent conversation entries.

**Parameters:**
- `k: int | None` - Number of entries to retrieve (default: uses `_recent_k`)

**Returns:**
- `list[Utterance]` - Recent conversation utterances

##### `get_recent_pe_history(k: int | None = None) -> list[PERecord]`
Retrieves the most recent PE history entries.

**Parameters:**
- `k: int | None` - Number of entries to retrieve (default: uses `_recent_k`)

**Returns:**
- `list[PERecord]` - Recent PE records

##### `get_recent_reflections(k: int | None = None) -> list[ReflectionRecord]`
Retrieves the most recent reflection entries.

**Parameters:**
- `k: int | None` - Number of entries to retrieve (default: uses `_recent_k`)

**Returns:**
- `list[ReflectionRecord]` - Recent reflections

##### `get_goal() -> Goal`
Returns the agent's goal.

**Returns:**
- `Goal` - The goal object

##### `get_last_pe() -> float`
Returns the last calculated PE value.

**Returns:**
- `float` - Last PE value, or 0.0 if no history exists

##### `_make_pre_act_value() -> str`
Formats memory data for pre_act context. Called automatically during `pre_act` phase.

**Returns:**
- `str` - Formatted string containing goal, recent conversation, PE history, and reflections

**Format:**
```
Goal: {goal.name}
Goal description: {goal.description}
Ideal value: {goal.ideal:.2f}

Recent conversation (last {recent_k}):
  [t={turn} {speaker}] {text}
  ...

Recent PE history:
  (turn {turn}) estimate={estimate:.2f}, PE={pe:+.2f} ← partner: "{partner_text}"
  ...

Recent reflections:
  (turn {turn}) {text}
  ...
```

##### `get_state() -> entity_component.ComponentState`
Serializes component state for checkpointing.

**Returns:**
- `dict` - State dictionary with:
  - `'conversation'`: List of utterance dicts
  - `'pe_history'`: List of PE record dicts
  - `'reflections'`: List of reflection dicts
  - `'goal'`: Goal dict
  - `'recent_k'`: Window size

##### `set_state(state: entity_component.ComponentState) -> None`
Restores component state from checkpoint.

**Parameters:**
- `state: dict` - State dictionary from `get_state()`

---

### 2. `PEEstimationComponent`

**Purpose:** Estimates current state and calculates PE during observation phase.

**Inheritance:**
- `action_spec_ignored.ActionSpecIgnored`
- `entity_component.ComponentWithLogging`

**Initialization:**
```python
def __init__(
    self,
    model: language_model.LanguageModel,
    memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
    pre_act_label: str = 'PE Estimation',
)
```

**Parameters:**
- `model: LanguageModel` - Language model for estimation
- `memory_component_key: str` - Key of PEMemoryComponent (default: 'PE_Memory')
- `pre_act_label: str` - Label for pre_act output (default: 'PE Estimation')

**Internal State:**
- `_model: LanguageModel` - Language model instance
- `_memory_component_key: str` - Key to access PEMemoryComponent
- `_last_estimate: float | None` - Last estimated state
- `_last_pe: float | None` - Last calculated PE
- `_last_partner_text: str` - Last extracted partner text

#### Methods

##### `pre_observe(observation: str) -> str`
Extracts partner's text from observation string. Called during `pre_observe` phase.

**Parameters:**
- `observation: str` - Raw observation string from game master

**Returns:**
- `str` - Empty string (no context to return)

**Behavior:**
- Parses observation to extract partner's actual message
- Handles multiple formats:
  - Quoted text: `"message"` or `'message'`
  - Agent name patterns: `Agent A: message` or `Partner: message`
  - Removes common prefixes: `Partner said:`, `Observation:`, etc.
  - Extracts text after other agent's name

**Example:**
```python
# Observation: 'Event: Agent B: Hello there!'
# Extracts: 'Hello there!'
```

##### `_make_pre_act_value() -> str`
Returns empty string (estimation component doesn't provide pre_act context).

**Returns:**
- `str` - Empty string

##### `post_observe() -> str`
Estimates state and computes PE. Called during `post_observe` phase.

**Returns:**
- `str` - Formatted result string: `'Estimated state: {estimate:.2f}, PE: {pe:+.2f}'`

**Behavior:**
1. Retrieves PEMemoryComponent and goal
2. Constructs LLM prompt asking for state estimation (0.0-1.0)
3. Parses LLM response to extract float value
4. Clamps estimate to [0.0, 1.0] range
5. Calculates PE = ideal - estimate
6. Stores PE record in memory
7. Logs result via logging channel

**LLM Prompt Format:**
```
You are {entity_name}. Goal: {goal.name}.
Goal description: {goal.description}
Ideal value on goal dimension: {goal.ideal:.2f}

Task: From only the partner's last response, estimate the CURRENT STATE on the goal dimension
as a single number in [0,1], where 1 means perfectly achieving the goal.
Partner said: "{partner_text}"

Respond with a single number in [0,1]. You may include a brief comment after the number.
```

##### `get_state() -> entity_component.ComponentState`
Serializes component state.

**Returns:**
- `dict` - State with:
  - `'last_estimate'`: Last estimated value
  - `'last_pe'`: Last PE value
  - `'last_partner_text'`: Last extracted partner text

##### `set_state(state: entity_component.ComponentState) -> None`
Restores component state.

**Parameters:**
- `state: dict` - State dictionary

---

### 3. `PEReflectionComponent`

**Purpose:** Generates reflections on how to reduce PE after estimation.

**Inheritance:**
- `action_spec_ignored.ActionSpecIgnored`
- `entity_component.ComponentWithLogging`

**Initialization:**
```python
def __init__(
    self,
    model: language_model.LanguageModel,
    memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
    pre_act_label: str = 'PE Reflection',
)
```

**Parameters:**
- `model: LanguageModel` - Language model for reflection generation
- `memory_component_key: str` - Key of PEMemoryComponent (default: 'PE_Memory')
- `pre_act_label: str` - Label for pre_act output (default: 'PE Reflection')

**Internal State:**
- `_model: LanguageModel` - Language model instance
- `_memory_component_key: str` - Key to access PEMemoryComponent
- `_last_reflection: str` - Last generated reflection text

#### Methods

##### `_make_pre_act_value() -> str`
Returns empty string (reflection component doesn't provide pre_act context).

**Returns:**
- `str` - Empty string

##### `post_observe() -> str`
Generates reflection after PE estimation. Called during `post_observe` phase.

**Returns:**
- `str` - Generated reflection text

**Behavior:**
1. Retrieves PEMemoryComponent, goal, and last PE value
2. Constructs LLM prompt asking for reflection on reducing PE
3. Generates reflection text via LLM
4. Stores reflection in memory
5. Logs reflection via logging channel

**LLM Prompt Format:**
```
You are {entity_name}. Goal: {goal.name}.
Given the PE of last turn (PE_last = {pe_last:+.3f}), write a short reflection:
What will you change next turn to REDUCE PE? Keep it concrete and brief.
```

##### `get_state() -> entity_component.ComponentState`
Serializes component state.

**Returns:**
- `dict` - State with `'last_reflection'`: Last reflection text

##### `set_state(state: entity_component.ComponentState) -> None`
Restores component state.

**Parameters:**
- `state: dict` - State dictionary

---

### 4. `PEActComponent`

**Purpose:** Generates utterances based on PE history and reflections.

**Inheritance:**
- `entity_component.ActingComponent`

**Initialization:**
```python
def __init__(
    self,
    model: language_model.LanguageModel,
    memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
)
```

**Parameters:**
- `model: LanguageModel` - Language model for action generation
- `memory_component_key: str` - Key of PEMemoryComponent (default: 'PE_Memory')

**Internal State:**
- `_model: LanguageModel` - Language model instance
- `_memory_component_key: str` - Key to access PEMemoryComponent

#### Methods

##### `get_action_attempt(
    context: entity_component.ComponentContextMapping,
    action_spec: entity_lib.ActionSpec,
) -> str`
Generates utterance based on PE context. Called during `act` phase.

**Parameters:**
- `context: ComponentContextMapping` - Context from all components (unused, gets data from memory directly)
- `action_spec: ActionSpec` - Action specification (unused)

**Returns:**
- `str` - Generated utterance text

**Behavior:**
1. Retrieves PEMemoryComponent, goal, and recent history
2. Formats recent conversation, PE history, and reflections
3. Constructs comprehensive LLM prompt with:
   - Goal definition
   - Recent conversation (last k turns)
   - Recent PE history
   - Recent reflections
4. Generates utterance via LLM
5. Stores utterance in memory
6. Returns utterance text

**LLM Prompt Format:**
```
You are {entity_name}. Your goal is "{goal.name}".
Definition: {goal.description}
Ideal value: {goal.ideal:.2f}

You must talk in a way that MINIMIZES PREDICTION ERROR (PE = ideal - estimated current state).
Consider recent conversation, PE history, and your reflections.

Recent conversation (last {recent_k}):
- [t={turn} {speaker}] {text}
- ...

Recent PE history:
- (turn {turn}) estimate={estimate:.2f}, PE={pe:+.2f} ← partner: "{partner_text}"
- ...

Recent reflections:
- (turn {turn}) {text}
- ...

Now produce ONE concise utterance to your partner that is likely to REDUCE PE next turn.
Avoid meta-talk; speak naturally.
```

##### `get_state() -> entity_component.ComponentState`
Returns empty state (act component is stateless).

**Returns:**
- `dict` - Empty dictionary

##### `set_state(state: entity_component.ComponentState) -> None`
No-op (act component is stateless).

**Parameters:**
- `state: dict` - Ignored

---

## Component Interaction Flow

### Turn Sequence

1. **Agent A Acts** (`PEActComponent.get_action_attempt`)
   - Generates utterance based on recent history
   - Stores utterance in `PEMemoryComponent`

2. **Agent B Observes** (`PEEstimationComponent.pre_observe`)
   - Extracts Agent A's text from observation
   - Stores in `_last_partner_text`

3. **Agent B Estimates** (`PEEstimationComponent.post_observe`)
   - Estimates current state from Agent A's text
   - Calculates PE = ideal - estimate
   - Stores PE record in `PEMemoryComponent`

4. **Agent B Reflects** (`PEReflectionComponent.post_observe`)
   - Generates reflection on reducing PE
   - Stores reflection in `PEMemoryComponent`

5. **Agent B Acts** (next turn, repeats from step 1)

### Data Flow

```
Observation → PEEstimationComponent.pre_observe()
                ↓
            Extract partner text
                ↓
            PEEstimationComponent.post_observe()
                ↓
            Estimate state → Calculate PE → Store in PEMemoryComponent
                ↓
            PEReflectionComponent.post_observe()
                ↓
            Generate reflection → Store in PEMemoryComponent
                ↓
            PEActComponent.get_action_attempt()
                ↓
            Retrieve history from PEMemoryComponent
                ↓
            Generate utterance → Store in PEMemoryComponent
```

---

## Usage Example

```python
from concordia.components.agent import pe_conversation as pe_components
from concordia.language_model import language_model

# Create goal
goal = pe_components.Goal(
    name="likability",
    description="Be perceived as likable by the partner (0=not liked, 1=fully liked). Aim for 1.0.",
    ideal=1.0
)

# Create components
memory = pe_components.PEMemoryComponent(goal=goal, recent_k=3)
estimation = pe_components.PEEstimationComponent(
    model=model,
    memory_component_key='PE_Memory'
)
reflection = pe_components.PEReflectionComponent(
    model=model,
    memory_component_key='PE_Memory'
)
act = pe_components.PEActComponent(
    model=model,
    memory_component_key='PE_Memory'
)

# Add to entity
entity.add_component('PE_Memory', memory)
entity.add_component('PE_Estimation', estimation)
entity.add_component('PE_Reflection', reflection)
entity.set_act_component(act)

# During simulation:
# - Observations trigger pre_observe/post_observe
# - Actions trigger get_action_attempt
# - All data stored in PEMemoryComponent
```

---

## Key Design Patterns

1. **Centralized Memory**: All conversation data stored in `PEMemoryComponent`
2. **Phase-Based Execution**: Components hook into specific phases (pre_observe, post_observe, act)
3. **LLM-Driven**: All estimation, reflection, and action generation uses language models
4. **State Management**: All components implement `get_state()`/`set_state()` for checkpointing
5. **Logging Integration**: Components use `_logging_channel` for debugging and analysis

---

## Notes

- Turn numbers are approximated from conversation length (may not be perfectly accurate)
- PE calculation: `PE = ideal - estimate` (positive PE means below ideal)
- All components are thread-safe when used within Concordia's phase system
- The `PEActComponent` ignores the `context` parameter and retrieves data directly from memory for clarity
