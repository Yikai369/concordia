# Conversation Memory and History Presentation

This document explains how conversation context is saved to memory and how history is presented to agents in both the original and standard implementations.

## Overview

Both implementations use a similar memory system:
1. **Storage**: Utterances are stored in a list (`conversation`) within each agent's memory
2. **Retrieval**: Recent conversation is retrieved using a sliding window (last `k` turns)
3. **Presentation**: History is formatted as text and included in LLM prompts

---

## Memory Storage

### Data Structure

**Original Implementation** (`pe_conversation_openai.py`):
```python
@dataclass
class AgentMemory:
    goal: Goal
    conversation: List[Utterance] = field(default_factory=list)  # Stores all utterances
    pe_history: List[PERecord] = field(default_factory=list)
    reflections: List[ReflectionRecord] = field(default_factory=list)
    # ... particle filter state ...
```

**Standard Implementation** (`impression_management_pe.py`):
```python
class IMPEMemoryComponent(PEMemoryComponent):
    def __init__(self, ...):
        self._conversation: list[Utterance] = []  # Stores all utterances
        self._pe_history: list[PERecord] = []
        self._reflections: list[ReflectionRecord] = []
        # ... particle filter state ...
```

### Utterance Structure

Both use the same `Utterance` dataclass:
```python
@dataclass
class Utterance:
    turn: int          # Turn number
    speaker: str       # Speaker name
    text: str          # Dialogue text
    body: str = ""     # Body language description
```

### When Utterances Are Saved

**Original Implementation**:

1. **Actor speaks** (line 609, 658):
   ```python
   utt = Utterance(turn=turn, speaker=self.name, text=text, body=body)
   self.memory.conversation.append(utt)
   ```

2. **Listener receives utterance** (line 739):
   ```python
   listener.memory.conversation.append(
       Utterance(turn=t, speaker=speaker.name, text=speaker_utt.text, body=speaker_utt.body)
   )
   ```

3. **Audience responds** (line 462):
   ```python
   utt = Utterance(turn=turn, speaker=self.name, text=dlg, body=body)
   self.memory.conversation.append(utt)
   ```

**Standard Implementation**:

1. **Actor speaks** (line 855):
   ```python
   memory.add_utterance(current_turn, self.get_entity().name, text, body)
   ```

2. **Audience responds** (line 466):
   ```python
   memory.add_utterance(current_turn, self.get_entity().name, dlg, body)
   ```

**Key Point**: Each agent maintains its own separate `conversation` list. When the actor speaks, the utterance is added to both:
- Actor's memory (when actor generates it)
- Audience's memory (when audience observes it)

---

## History Retrieval

### Sliding Window Approach

Both implementations use a **sliding window** to retrieve recent conversation:

**Original Implementation** (line 379-385):
```python
def recent_conversation(self, k: Optional[int] = None) -> List[Utterance]:
    """Return the last `k` Utterance objects from memory (most recent last)."""
    k = k if k is not None else self.recent_k
    return self.memory.conversation[-k:] if self.memory.conversation else []
```

**Standard Implementation** (inherited from `PEMemoryComponent`, line 113-117):
```python
def get_recent_conversation(self, k: int | None = None) -> list[Utterance]:
    """Get recent conversation entries."""
    if k is None:
        k = self._recent_k
    return self._conversation[-k:]
```

**How it works**:
- `recent_k` defaults to 3 (configurable via `--window` argument)
- `[-k:]` gets the last `k` items from the list
- Returns most recent `k` utterances, ordered chronologically

**Example**:
- If `conversation = [utt1, utt2, utt3, utt4, utt5]` and `k=3`
- Returns `[utt3, utt4, utt5]` (last 3 utterances)

---

## History Formatting

### Format Function

Both implementations format conversation history as text for LLM prompts:

**Original Implementation** (line 387-395):
```python
def format_conversation(self, conv: List[Utterance]) -> str:
    """Format a list of Utterances into a compact, readable block."""
    if not conv:
        return "- (none)"
    return chr(10).join(f"- [t={u.turn} {u.speaker}] {u.text}" for u in conv)
```

**Standard Implementation** (line 215-221):
```python
def format_conversation(self, utterances: list[Utterance]) -> str:
    """Format conversation for prompts."""
    if not utterances:
        return '- (none)'
    return '\n'.join(
        f'- [t={u.turn} {u.speaker}] {u.text}' for u in utterances
    )
```

**Output Format**:
```
- [t=1 John] I effectively bridge communication between teams.
- [t=2 Jane] The candidate shows potential in technical understanding.
- [t=3 John] My approach emphasizes close collaboration.
```

**Note**: Body language is **not** included in the formatted conversation history, only the dialogue text.

---

## History Presentation in Prompts

### How History is Included

History is retrieved, formatted, and inserted into LLM prompts in several places:

#### 1. **Actor Acting (Subsequent Turns)**

**Original Implementation** (line 640-642):
```python
conv_k = self.recent_conversation(self.recent_k)
# ...
prompt = f"""...
Recent conversation (last {self.recent_k}):
{self.format_conversation(conv_k)}
..."""
```

**Standard Implementation** (line 816, 834):
```python
conv_k = memory.get_recent_conversation()
# ...
prompt = f"""...
Recent conversation (last {recent_k}):
{memory.format_conversation(conv_k)}
..."""
```

#### 2. **Audience Responding**

**Original Implementation** (line 434, 442-443):
```python
conv_k = self.recent_conversation(self.recent_k)
# ...
resp_prompt = f"""...
Recent conversation (last {self.recent_k}):
{self.format_conversation(conv_k)}
..."""
```

**Standard Implementation** (line 444, 451-452):
```python
conv_k = memory.get_recent_conversation()
# ...
resp_prompt = f"""...
Recent conversation (last {memory._recent_k}):
{memory.format_conversation(conv_k)}
..."""
```

### Complete Prompt Structure

Here's how a typical prompt looks with history included:

```
CULTURAL NORMS YOU FOLLOW:
- Norm1: Description1
- Norm2: Description2

YOUR PERSONALITY TRAITS:
- Trait1 (2/3): Assertion1
- Trait2 (1/3): Assertion2

You are John. You want to achieve: competence.
Definition: Be perceived as competent...
Ideal value: 1.00

Current belief about the interviewer's evaluation = 0.75 (on a scale from 0-1).

Recent conversation (last 3):
- [t=1 John] I effectively bridge communication between teams.
- [t=2 Jane] The candidate shows potential in technical understanding.
- [t=3 John] My approach emphasizes close collaboration.

Recent I_hat (belief) history:
- (turn 1) I_hat=0.50
- (turn 2) I_hat=0.65
- (turn 3) I_hat=0.75

Recent reflections:
- (turn 2) I will focus on demonstrating technical skills.
- (turn 3) I will emphasize collaboration examples.

Produce a short utterance...
```

---

## Key Differences Between Implementations

### 1. **Memory Component Location**

- **Original**: Memory is a dataclass (`AgentMemory`) stored directly in the `Agent` class
- **Standard**: Memory is a Concordia component (`IMPEMemoryComponent`) attached to entities

### 2. **Access Pattern**

- **Original**: Direct access via `self.memory.conversation`
- **Standard**: Access via component: `memory.get_recent_conversation()`

### 3. **Storage Method**

- **Original**: Direct list append: `self.memory.conversation.append(utt)`
- **Standard**: Method call: `memory.add_utterance(turn, speaker, text, body)`

### 4. **Synchronization**

- **Original**: Manual synchronization - orchestrator explicitly adds utterances to listener's memory (line 739)
- **Standard**: Automatic via Concordia's observation system - when audience observes, it extracts the utterance

---

## Memory Lifecycle Example

Let's trace a complete turn to see how memory works:

### Turn 2 Example

**Step 1: Actor speaks**
- Actor generates utterance: "I emphasize collaboration."
- Actor's memory: `conversation.append(Utterance(turn=2, speaker="John", text="I emphasize collaboration.", body="Maintains eye contact"))`
- Actor's conversation list: `[utt_turn1, utt_turn2]`

**Step 2: Audience observes**
- Audience receives actor's utterance
- Audience's memory: `conversation.append(Utterance(turn=2, speaker="John", text="I emphasize collaboration.", body="Maintains eye contact"))`
- Audience's conversation list: `[utt_turn1, utt_turn2]`

**Step 3: Audience responds**
- Audience retrieves recent conversation: `get_recent_conversation(k=3)` → `[utt_turn1, utt_turn2]`
- Audience formats it: `format_conversation([utt_turn1, utt_turn2])` → `"- [t=1 John] ...\n- [t=2 John] ..."`
- Audience includes in prompt and generates response
- Audience's memory: `conversation.append(Utterance(turn=2, speaker="Jane", text="Good approach.", body="Nods"))`
- Audience's conversation list: `[utt_turn1, utt_turn2_actor, utt_turn2_audience]`

**Step 4: Actor observes audience response**
- Actor receives audience's utterance
- Actor's memory: `conversation.append(Utterance(turn=2, speaker="Jane", text="Good approach.", body="Nods"))`
- Actor's conversation list: `[utt_turn1, utt_turn2_actor, utt_turn2_audience]`

**Step 5: Actor acts (next turn)**
- Actor retrieves recent conversation: `get_recent_conversation(k=3)` → `[utt_turn1, utt_turn2_actor, utt_turn2_audience]`
- Actor formats it and includes in prompt
- Actor generates next utterance based on this history

---

## Important Notes

### 1. **Separate Memory Per Agent**

Each agent has its own `conversation` list. They don't share memory directly - utterances are copied when agents observe each other.

### 2. **Sliding Window**

Only the last `k` utterances are included in prompts. Older utterances are still in memory but not used in prompts (unless `k` is increased).

### 3. **Body Language Not in History**

The formatted conversation history only includes dialogue text, not body language. Body language is:
- Stored in the `Utterance` object
- Used when generating responses (in prompts)
- But not shown in the "Recent conversation" section

### 4. **Turn Numbers**

Turn numbers are preserved in the `Utterance` objects, allowing agents to see the chronological order and turn context.

### 5. **Memory Persistence**

Memory persists across turns within a single conversation run. When the conversation ends, memory is typically discarded unless explicitly saved (e.g., to JSON logs).

---

## Summary

1. **Storage**: Utterances are stored in a list (`conversation`) in each agent's memory
2. **Retrieval**: Recent conversation is retrieved using `get_recent_conversation(k)` which returns the last `k` utterances
3. **Formatting**: History is formatted as text with format `"- [t={turn} {speaker}] {text}"`
4. **Presentation**: Formatted history is included in LLM prompts under "Recent conversation (last {k}):"
5. **Separation**: Each agent maintains its own memory, and utterances are copied when agents observe each other

This design allows agents to have context-aware conversations while keeping memory usage bounded by the sliding window size.





