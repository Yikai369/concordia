# PE Conversation Implementation Plan for Concordia

## Overview
Implement the PE (Prediction Error) conversation system in Concordia framework, where two agents adapt their conversation based on prediction errors.

## Original System Analysis

### Key Components:
1. **Agent Memory**: Stores conversation, PE history, reflections, and goal
2. **Observe Phase**: Estimates current state from partner's response, computes PE
3. **Learning Phase**: Generates reflection to reduce PE
4. **Act Phase**: Produces utterance based on conversation, PE history, and reflections

### Turn Flow:
1. Speaker acts (generates utterance)
2. Listener observes (estimates state, computes PE)
3. Listener learns (generates reflection)
4. Log turn data

## Concordia Implementation Plan

### 1. Custom Components

#### 1.1 PEMemoryComponent
- **Purpose**: Store conversation history, PE records, reflections, and goal
- **Type**: ContextComponent (ActionSpecIgnored)
- **State**:
  - `conversation`: List of utterances (turn, speaker, text)
  - `pe_history`: List of PE records (turn, partner_text, estimate, pe)
  - `reflections`: List of reflection records (turn, text)
  - `goal`: Goal object (name, description, ideal)
  - `recent_k`: Window size for recent history
- **Methods**:
  - `add_utterance(turn, speaker, text)`: Add conversation entry
  - `add_pe_record(turn, partner_text, estimate, pe)`: Add PE record
  - `add_reflection(turn, text)`: Add reflection
  - `get_recent_conversation(k)`: Get last k conversation entries
  - `get_recent_pe_history(k)`: Get last k PE records
  - `get_recent_reflections(k)`: Get last k reflections
  - `get_state()` / `set_state()`: For checkpointing

#### 1.2 PEEstimationComponent
- **Purpose**: Estimate current state and compute PE during observe phase
- **Type**: ContextComponent (ActionSpecIgnored)
- **Dependencies**: PEMemoryComponent, LanguageModel
- **Methods**:
  - `pre_observe(observation)`: Extract partner's text from observation
  - `post_observe()`: Call LLM to estimate state, compute PE, store in memory
- **Logic**:
  - Parse observation to extract partner's utterance
  - Prompt LLM: "Estimate current state on goal dimension [0,1]"
  - Parse float from response
  - Compute PE = ideal - estimate
  - Store in PEMemoryComponent

#### 1.3 PEReflectionComponent
- **Purpose**: Generate reflection to reduce PE
- **Type**: ContextComponent (ActionSpecIgnored)
- **Dependencies**: PEMemoryComponent, LanguageModel
- **Methods**:
  - `post_observe()`: After PE estimation, generate reflection
- **Logic**:
  - Get last PE from memory
  - Prompt LLM: "What will you change next turn to REDUCE PE?"
  - Store reflection in PEMemoryComponent

#### 1.4 PEActComponent
- **Purpose**: Generate utterance based on PE history, reflections, and conversation
- **Type**: ActingComponent
- **Dependencies**: PEMemoryComponent, LanguageModel
- **Methods**:
  - `get_action_attempt(context, action_spec)`: Generate utterance
- **Logic**:
  - Get recent conversation, PE history, reflections from memory
  - Format context for LLM
  - Prompt LLM: "Produce utterance to REDUCE PE"
  - Store utterance in memory
  - Return utterance

### 2. Custom Prefabs

#### 2.1 PEEntity Prefab
- **Components**:
  - Instructions (goal description)
  - PEMemoryComponent
  - PEEstimationComponent
  - PEReflectionComponent
  - PEActComponent (as act_component)
  - ObservationToMemory (standard)
- **Parameters**:
  - `name`: Entity name
  - `goal_name`: Goal name (e.g., "likability")
  - `goal_description`: Goal description
  - `goal_ideal`: Ideal value (default 1.0)
  - `recent_k`: Window size (default 3)

#### 2.2 PEConversationGameMaster Prefab
- **Purpose**: Orchestrate PE conversation flow
- **Components**:
  - Instructions
  - PlayerCharacters
  - NextActing (alternating between two agents)
  - NextActionSpec
  - MakeObservation (passes partner's utterance)
  - EventResolution (stores utterance as event)
  - Terminate (after max turns)
- **Special Logic**:
  - Alternates between two agents (fixed order)
  - After each action, triggers observe->estimate->reflect cycle
  - Logs turn data

### 3. Simulation Structure

#### 3.1 Main Script Structure
```python
# Setup
- Load language model
- Load embedder
- Load prefabs

# Configuration
- Create two PEEntity instances
- Create PEConversationGameMaster instance
- Create initializer

# Run
- Initialize simulation
- Run with max_steps = total_turns * 2 (each turn = 2 steps: act + observe)
- Extract log data
- Save to JSON
```

#### 3.2 Turn Flow in Concordia
1. **Step 1 (Act)**: Game master selects speaker, speaker acts
2. **Step 2 (Observe)**: Game master creates observation for listener
   - Listener's PEEstimationComponent processes observation
   - Listener's PEReflectionComponent generates reflection
3. **Repeat** for next turn

### 4. Data Extraction

#### 4.1 Log Structure
- Access `raw_log` from simulation
- Extract per-turn data:
  - Turn number
  - Speaker name
  - Listener name
  - Speaker text
  - Listener estimate
  - Listener PE
  - Listener reflection

#### 4.2 JSON Output
- Format similar to original: TurnLog dataclass
- Save to JSON file

### 5. Implementation Files

1. **components/pe_conversation.py**: Custom components
2. **prefabs/entity/pe_entity.py**: PEEntity prefab
3. **prefabs/game_master/pe_conversation.py**: Game master prefab
4. **examples/pe_conversation_concordia.py**: Main script (.py)
5. **examples/pe_conversation_concordia.ipynb**: Notebook version
6. **tests/test_pe_conversation.py**: Unit tests

### 6. Key Challenges & Solutions

#### Challenge 1: Triggering PE estimation after observation
- **Solution**: Use `post_observe()` phase in PEEstimationComponent
- **Alternative**: Custom game master that explicitly calls estimation

#### Challenge 2: Alternating between agents
- **Solution**: Use `NextActingInFixedOrder` with alternating sequence

#### Challenge 3: Extracting partner's text from observation
- **Solution**: Parse observation string to extract last utterance
- **Alternative**: Game master formats observation with clear markers

#### Challenge 4: Storing turn-specific data
- **Solution**: Track current turn in game master, pass to components via observation

### 7. Testing Strategy

1. **Unit Tests**:
   - PEMemoryComponent: Add/retrieve data
   - PEEstimationComponent: Parse estimate from LLM response
   - PEReflectionComponent: Generate reflection
   - PEActComponent: Generate utterance with context

2. **Integration Tests**:
   - Full conversation with 2 agents, 2 turns
   - Verify PE calculation
   - Verify reflection generation
   - Verify log output

3. **End-to-End Tests**:
   - Run full simulation
   - Compare output format with original
   - Verify JSON structure

## Implementation Order

1. Create PEMemoryComponent (foundation)
2. Create PEEstimationComponent
3. Create PEReflectionComponent
4. Create PEActComponent
5. Create PEEntity prefab
6. Create PEConversationGameMaster prefab
7. Create main script (.py)
8. Create notebook (.ipynb)
9. Write unit tests
10. Test and debug
11. Refine and optimize
