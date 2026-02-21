# PE Conversation System: Comparison Document

## Overview

This document provides a detailed comparison between two implementations of the PE (Prediction Error) Conversation System:

- **`pe_conversation_openai.py`**: Standard/clean implementation
- **`pe_conversation_prototype.py`**: Prototype with experimental features

Both systems implement the same core architecture (particle filter-based belief tracking with prediction error learning) but differ significantly in their feature sets, world-building approach, and implementation details.

---

## Table of Contents

1. [Quick Comparison Summary](#quick-comparison-summary)
2. [Architectural Differences](#architectural-differences)
3. [Data Structure Differences](#data-structure-differences)
4. [Cultural Norms & World-Building](#cultural-norms--world-building)
5. [Personality Traits Handling](#personality-traits-handling)
6. [Agent Methods Comparison](#agent-methods-comparison)
7. [Interview Context Differences](#interview-context-differences)
8. [Output & Logging Differences](#output--logging-differences)
9. [Code Quality & Maintainability](#code-quality--maintainability)
10. [Use Case Recommendations](#use-case-recommendations)

---

## Quick Comparison Summary

| Feature | `pe_conversation_openai.py` | `pe_conversation_prototype.py` |
|---------|----------------------------|-------------------------------|
| **Complexity** | Simpler, cleaner | More complex, experimental |
| **World-Building** | Direct, minimal | Elaborate "2A25" fictional world |
| **Cultural Norms** | Simple list | World-building narrative |
| **Personality Traits** | Score-based (0-3) | LLM-generated paragraphs |
| **Self-Reflection** | Basic | Advanced with multiple methods |
| **Option Generation** | No | Yes (4-option selection) |
| **Spreadsheet Support** | No | Yes (Excel trait extraction) |
| **Interview Role** | Product Manager | Customer Service Agent |
| **Agent Names** | John, Jane | Riffer, Caden |
| **Question Checks** | No | Yes (personality/context) |
| **Code Comments** | Standard | More debug prints |

---

## Architectural Differences

### Core Architecture

Both implementations share the same fundamental architecture:

```
ACT → OBSERVE → UPDATE → LEARN → LOG
```

However, the prototype adds additional layers:

**Standard (`pe_conversation_openai.py`):**
```
Actor acts → Audience evaluates → Actor updates belief → Actor reflects
```

**Prototype (`pe_conversation_prototype.py`):**
```
Actor acts → [Self-reflection] → Audience evaluates → [Self-reflection] →
Actor updates belief → Actor reflects → [Question checks]
```

### Key Architectural Additions in Prototype

1. **Self-Reflection Loops**: Both actor and audience can reflect on their own responses
2. **Option Generation**: Audience can generate multiple response options and choose
3. **Question Checks**: Agents verify their understanding of personality and context
4. **Trait Paragraph Generation**: Traits are converted to narrative paragraphs via LLM

---

## Data Structure Differences

### Utterance Class

**Standard:**
```python
@dataclass
class Utterance:
    turn: int
    speaker: str      # ← Uses "speaker"
    text: str
    body: str = ""
```

**Prototype:**
```python
@dataclass
class Utterance:
    turn: int
    actor: str        # ← Uses "actor"
    text: str
    body: str = ""
```

**Impact**: The prototype uses role-based naming (`actor`/`audience`) while the standard uses generic `speaker`/`listener`. This affects all conversation formatting methods.

### PersonalityTrait Class

**Standard:**
```python
@dataclass
class PersonalityTrait:
    name: str
    assertion: str
    negative_assertion: Optional[str] = None
```

**Prototype:**
```python
@dataclass
class PersonalityTrait:
    survey: str       # ← Different field name
    assertion: str
```

**Impact**:
- Standard supports both positive and negative assertions
- Prototype uses `survey` field (likely for spreadsheet organization)
- Prototype doesn't have `negative_assertion` field

### TurnLog Class

**Standard:**
```python
@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str
    listener: str
    speaker_text: str
    speaker_body: str
    audience_I: float
    audience_text: str      # Single response
    audience_body: str
    actor_I_hat: float
    actor_pe: float
    ess: float
```

**Prototype:**
```python
@dataclass
class TurnLog:
    time: str
    turn: int
    actor: str
    audience: str
    actor_text: str
    actor_body: str
    audience_I: float
    audience_text0: str     # Initial response
    audience_body0: str
    audience_text: str      # Final response (after reflection)
    audience_body: str
    actor_I_hat: float
    actor_pe: float
    actor_personality_check: str    # Additional fields
    actor_context_check: str
    audience_personality_check: str
    audience_context_check: str
    ess: float
```

**Key Differences:**
- Prototype tracks both initial and final audience responses (before/after self-reflection)
- Prototype includes question check responses
- Different field naming conventions

---

## Cultural Norms & World-Building

### Standard Implementation

**Approach**: Direct, minimal cultural norms

```python
def _prompt_header(self) -> str:
    norms_text = "\n".join(
        f"- {n.name}: {n.description}" for n in self.cultural_norms
    ) + "\n\n"
    # Simple, direct prompt
    return f"You are {self.name}. {self.memory.goal.description}..."
```

**Characteristics:**
- Straightforward norm listing
- No fictional world-building
- Direct instructions to agents

### Prototype Implementation

**Approach**: Elaborate fictional world "2A25"

```python
def _prompt_header(self) -> str:
    norms_prompt = f"""You are initializing a social simulation of an alternate
    reality called 2A25. The dominant cognitive-cultural order is the Cadens
    majority who all follow these cultural norms: {norms_text}.

    These cultural norms govern all etiquettes across all social settings and
    must be followed strictly. Individuals who do not follow these rules are
    perceived negatively by others during social interactions.

    In 2A25, there is another social minority group called the Riffers. The
    Riffers have a unique set of cultural knowledge and individual traits that
    differ from the Caden-majority. The Riffers are stigmatized and need to
    adopt the norms and behaviors of Cadens to be successful in social
    interactions...

    This setting is a fictional social world. This world is not an allegory for
    any real-world group...
    """
```

**Characteristics:**
- Extensive world-building narrative
- Fictional groups (Cadens, Riffers)
- Stigma and social dynamics
- Explicit disclaimers about fictional nature
- More immersive but also more complex

**Use Case**: The prototype's approach is designed for studying:
- Stigma and social exclusion
- Cultural norm learning
- Minority group adaptation
- Social simulation research

---

## Personality Traits Handling

### Standard Implementation

**Method**: Score-based trait system (Scores + Assertions)

**Key Point**: Uses **numeric scores (0-3) combined with assertion statements**

```python
# 1. Generate scores for each trait
aud_trait_scores = generate_trait_scores(rng, traits, is_audience=True)  # Scores 2-3
actor_trait_scores = generate_trait_scores(rng, traits, is_audience=False)  # Scores 0-1

# 2. Pass scores to agent
agent = Agent(
    traits=traits,
    trait_scores=actor_trait_scores,  # ← Scores dictionary provided
)

# 3. Format in prompts with scores
def _prompt_header(self) -> str:
    if self.traits:
        traits_text = "YOUR PERSONALITY TRAITS, scored from 0 to 3:\n" + "\n".join(
            f"- {t.name} ({self.trait_scores.get(t.name, 'NA')} / 3): {t.assertion}"
            for t in self.traits
        ) + "\n\n"
    return traits_text
```

**Characteristics:**
- ✅ **Uses scores (0-3)** for each trait
- ✅ **Combines scores with assertion statements** in prompts
- ✅ Direct, explicit formatting
- ✅ Simple, transparent
- ✅ Easy to understand and modify
- ✅ No additional LLM calls needed

**Example Prompt Output:**
```
YOUR PERSONALITY TRAITS, scored from 0 to 3:
- Detail-focused (2 / 3): I tend to focus on individual parts...
- Avoids eye contact (3 / 3): I do not make eye contact...
- Not social (1 / 3): I do not enjoy social situations...
```

### Prototype Implementation

**Method**: Statement-only system (No Scores) → Converted to Paragraphs

**Key Point**: Uses **assertion statements only (no scores)**, then converts to narrative paragraphs via LLM

```python
# 1. Extract traits (statements only, no scores)
aud_traits = extract_traits_from_spreadsheet("autism-measures-compilation.xlsx")

# 2. Pass to agent WITHOUT scores
agent = Agent(
    traits=aud_traits,
    trait_scores=None,  # ← No scores! Only statements
)

# 3. Initialize: Convert statements to paragraph via LLM
def initialize_personality_traits(self, traits: List[str]) -> None:
    trait_list = "\n".join(f"- {s.assertion}" for s in traits)  # ← Only assertions
    prompt = f"Write a detailed paragraph describing this person based on statements..."
    traits_paragraph = self.llm(prompt)  # ← LLM generates paragraph
    self.trait_paragraph = traits_paragraph

# 4. Use paragraph in prompts (not individual traits)
def _prompt_header(self) -> str:
    if self.trait_paragraph:
        traits_prompt = f"""The following paragraph describes {self.name} interactions
                           and how they perceive, process, and interact with the social
                           world.{self.trait_paragraph}"""
```

**Characteristics:**
- ❌ **No scores** - `trait_scores=None` always
- ✅ **Uses assertion statements only** (from traits)
- ✅ **Converts statements to narrative paragraphs** via LLM
- ✅ More natural, flowing descriptions
- ⚠️ Less explicit (scores not visible)
- ⚠️ Requires additional LLM call per agent (cost/time)
- ⚠️ Stored in `trait_paragraph` attribute

**Example Prompt Output:**
```
The following paragraph describes Riffer interactions and how they perceive,
process, and interact with the social world. This person tends to focus
intensely on specific details rather than seeing the broader picture. They
avoid making eye contact during conversations, which is a consistent
behavioral pattern. Their interactions are characterized by a preference
for structured, predictable environments...
```

**Critical Difference Summary:**

| Aspect | Standard | Prototype |
|--------|----------|-----------|
| **Trait Format** | **Scores (0-3) + Assertions** | **Assertions Only (No Scores)** |
| **Score Generation** | ✅ `generate_trait_scores()` | ❌ Not used (`trait_scores=None`) |
| **Prompt Format** | `"- {name} ({score}/3): {assertion}"` | Narrative paragraph |
| **Initialization** | Direct use in prompts | LLM converts to paragraph |
| **Agent Parameter** | `trait_scores={...}` | `trait_scores=None` |

**Trade-offs:**

| Aspect | Score-Based (Standard) | Paragraph-Based (Prototype) |
|--------|----------------------|----------------------------|
| **Explicitness** | High | Medium |
| **LLM Calls** | 0 | 1 per agent |
| **Cost** | Lower | Higher |
| **Naturalness** | Lower | Higher |
| **Reproducibility** | High | Lower (LLM variance) |
| **Debugging** | Easy | Harder |

---

## Agent Methods Comparison

### Methods Present in Both

| Method | Standard | Prototype | Notes |
|--------|----------|-----------|-------|
| `act()` | ✅ | ✅ | Initial utterance generation |
| `act_based_on_belief()` | ✅ | ✅ | Belief-conditioned utterance |
| `audience_evaluate_and_respond()` | ✅ | ✅ | Evaluation and response |
| `actor_update_particles()` | ✅ | ✅ | Particle filter update |
| `learning()` | ✅ | ✅ | Reflection generation |
| `_prompt_header()` | ✅ | ✅ | Prompt construction |
| `format_response()` | ✅ | ✅ | Parse DIALOGUE/BODY |

### Methods Only in Prototype

#### 1. `initialize_personality_traits()`

**Purpose**: Convert trait list to narrative paragraph via LLM

```python
def initialize_personality_traits(self, traits: List[str]) -> None:
    # Generates trait paragraph via LLM
    traits_paragraph = self.llm(prompt)
    self.trait_paragraph = traits_paragraph
    return traits_paragraph
```

**Impact**: Adds initialization step, increases LLM calls

#### 2. `question_check()`

**Purpose**: Verify agent's understanding of personality and context

```python
def question_check(self):
    context_check = "What kind of situation is this? Summarize..."
    personality_check = "What kind of person are you? How does..."

    context_response = self.llm(context_check)
    personality_response = self.llm(personality_check)

    return context_response, personality_response
```

**Impact**:
- Adds 2 LLM calls per turn per agent
- Useful for debugging/verification
- Increases cost and runtime

#### 3. `generate_option_space()`

**Purpose**: Generate 4 distinct response options

```python
def generate_option_space(self, prompt) -> List[Tuple[str, str]]:
    options_prompt = """Generate a set of 4 distinct options for how to
                       respond next in the conversation..."""
    raw_output = self.llm(prompt + options_prompt)
    # Parse and return 4 options
    return options
```

**Impact**:
- Enables multi-option selection
- More sophisticated response generation
- Additional LLM call

#### 4. `choose_option()`

**Purpose**: Select one option from generated set

```python
def choose_option(self, options: List[Tuple[str, str]]) -> Tuple[str, str]:
    choice_prompt = """Choose one of the four options..."""
    raw_output = self.llm(choice_prompt)
    # Extract choice index
    return options[choice_idx]
```

**Impact**:
- Adds deliberation step
- More controlled response selection
- Additional LLM call

#### 5. `actor_self_reflection()`

**Purpose**: Actor reflects on own response for trait alignment

```python
def actor_self_reflection(self, actor_utt: Utterance, aud_utt: Utterance) -> str:
    critique_prompt = f"""You just replied... Is it consistent with who
                          you are and what you know about the world?"""
    critique = self.llm(critique_prompt)
    return critique
```

**Impact**:
- Used in `act_based_on_belief()` if traits enabled
- Improves trait consistency
- Additional LLM call

#### 6. `audience_self_reflection()`

**Purpose**: Audience reflects on own response for norm/trait alignment

```python
def audience_self_reflection(self, actor_utt: Utterance,
                            audience_reply: Utterance, I_t: float) -> str:
    critique_prompt = f"""You just replied... Write an improved response
                          to align with your cultural norms and personality
                          traits..."""
    critique = self.llm(critique_prompt)
    return critique
```

**Impact**:
- Used in `audience_evaluate_and_respond()` if norms/traits enabled
- Improves norm/trait consistency
- Additional LLM call
- Prototype tracks both initial and final responses

### Method Signature Differences

#### `audience_evaluate_and_respond()`

**Standard:**
```python
def audience_evaluate_and_respond(self, turn: int, actor_utt: Utterance)
    -> Tuple[float, Utterance]:
    # Returns: (I_t, utterance)
```

**Prototype:**
```python
def audience_evaluate_and_respond(self, turn: int, actor_utt: Utterance)
    -> Tuple[float, Utterance]:
    # Returns: (I_t, utterance)
    # But internally may call self_reflection and track both responses
```

**Note**: Prototype's implementation may return different values depending on self-reflection, but signature is the same.

#### `act_based_on_belief()`

**Standard:**
```python
def act_based_on_belief(self, turn: int, belief: float,
                        audience_last_utt: Utterance) -> Utterance:
    # Simple implementation
```

**Prototype:**
```python
def act_based_on_belief(self, turn: int, belief: float,
                        audience_last_utt: Utterance) -> Utterance:
    # May call actor_self_reflection() if traits enabled
    if self.traits:
        new_resp_raw = self.actor_self_reflection(utt, audience_last_utt)
        # Use improved response
```

---

## Interview Context Differences

### Standard Implementation

**Role**: Product Manager

```python
interview_role = """
Role: Product Manager

Responsibilities:
- Defining product requirements
- Making tradeoff decisions
- Communicating priorities clearly

The interview evaluates:
- Structured thinking
- Decision rationale
- Clarity of communication
"""
```

**Agent Setup:**
- Actor: "John" (interviewee)
- Audience: "Jane" (interviewer)
- Actor has NO cultural norms
- Audience has cultural norms (if enabled)

### Prototype Implementation

**Role**: Customer Service Agent

```python
interview_role = """
Role: Customer Service Agent

Responsibilities:
- Understanding customer issues and needs
- Resolving problems efficiently and accurately
- Communicating solutions clearly and empathetically

The interview evaluates:
- Problem-solving ability
- Judgment and decision-making under constraints
- Clarity and effectiveness of communication
"""
```

**Agent Setup:**
- Actor: "Riffer" (interviewee, stigmatized minority)
- Audience: "Caden" (interviewer, majority)
- **Both** have cultural norms (Riffer needs to learn them)
- Includes question banks for interviewer/interviewee

**Question Banks:**

**For Caden (Interviewer):**
```
- Tell me about your customer service experience
- What does a good support interaction look like to you?
- Walk me through how you handle an angry customer.
- ...
```

**For Riffer (Interviewee):**
```
- Experience 1: Managed high-volume frontline support...
- Experience 2: Resolved billing disputes...
- Experience 3: Delivered ticket support...
- Experience 4: Improved support processes...
```

**Key Difference**: Prototype includes specific experience examples and question prompts, making it more structured for the Customer Service role.

---

## Output & Logging Differences

### JSON Output Structure

**Standard:**
```json
[
  {
    "time": "2025-12-27T23:04:37Z",
    "turn": 1,
    "speaker": "John",
    "listener": "Jane",
    "speaker_text": "...",
    "speaker_body": "...",
    "audience_I": 0.8,
    "audience_text": "...",
    "audience_body": "...",
    "actor_I_hat": 0.75,
    "actor_pe": 0.15,
    "ess": 150.5
  }
]
```

**Prototype:**
```json
{
  "actor_traits": "Generated trait paragraph...",
  "audience_traits": "Generated trait paragraph...",
  "turns": [
    {
      "time": "2025-12-27T23:04:37Z",
      "turn": 1,
      "actor": "Riffer",
      "audience": "Caden",
      "actor_text": "...",
      "actor_body": "...",
      "audience_I": 0.8,
      "audience_text0": "...",      // Initial response
      "audience_body0": "...",
      "audience_text": "...",       // Final response
      "audience_body": "...",
      "actor_I_hat": 0.75,
      "actor_pe": 0.15,
      "actor_personality_check": "...",
      "actor_context_check": "...",
      "audience_personality_check": "...",
      "audience_context_check": "...",
      "ess": 150.5
    }
  ]
}
```

**Key Differences:**
- Prototype includes trait paragraphs at top level
- Prototype tracks both initial and final audience responses
- Prototype includes question check responses
- Different field naming (speaker/listener vs actor/audience)

### Visualization

**Standard:**
- Generates 3 plots: `pe.png`, `delta_I.png`, `learning_gain.png`
- Always enabled

**Prototype:**
- Plotting code exists but is commented out
- `plot_learning_dynamics(runlog, save_dir=study.save_dir)` is disabled

---

## Code Quality & Maintainability

### Standard Implementation

**Strengths:**
- ✅ Clean, readable code
- ✅ Consistent naming conventions
- ✅ Minimal dependencies
- ✅ Well-structured methods
- ✅ Good separation of concerns
- ✅ Easy to understand and modify

**Weaknesses:**
- ⚠️ Less experimental features
- ⚠️ Simpler trait handling

### Prototype Implementation

**Strengths:**
- ✅ Rich experimental features
- ✅ Advanced self-reflection mechanisms
- ✅ Option generation for better responses
- ✅ Question checks for verification
- ✅ Spreadsheet support for trait extraction

**Weaknesses:**
- ⚠️ More complex codebase
- ⚠️ Debug print statements left in code
- ⚠️ More LLM calls (higher cost)
- ⚠️ Harder to understand flow
- ⚠️ Requires pandas for spreadsheet support
- ⚠️ Some commented-out code sections

### Code Examples

**Standard - Clean Method:**
```python
def format_conversation(self, conv: List[Utterance]) -> str:
    if not conv:
        return "- (none)"
    return chr(10).join(f"- [t={u.turn} {u.speaker}] {u.text}" for u in conv)
```

**Prototype - With Debug:**
```python
def format_conversation(self, conv: List[Utterance]) -> str:
    if not conv:
        return "- (none)"
    return chr(10).join(f"- [t={u.turn} {u.actor}] {u.text}" for u in conv)

# Later in code:
print(resp_prompt)  # ← Debug print
print(prompt)       # ← Debug print
```

---

## Use Case Recommendations

### Use `pe_conversation_openai.py` (Standard) When:

1. **General Research**: You need a clean, well-documented implementation
2. **Cost Sensitivity**: You want to minimize LLM API calls
3. **Reproducibility**: You need consistent, score-based trait handling
4. **Maintainability**: You want easy-to-modify code
5. **Product Manager Studies**: You're studying PM interview contexts
6. **Learning/Teaching**: You're learning the system architecture
7. **Production Use**: You need stable, tested code

### Use `pe_conversation_prototype.py` When:

1. **Stigma Research**: You're studying social exclusion and adaptation
2. **Cultural Learning**: You need to model norm learning dynamics
3. **Advanced Features**: You need self-reflection and option generation
4. **Customer Service Context**: You're studying CS agent interviews
5. **Trait Paragraphs**: You prefer narrative trait descriptions
6. **Question Verification**: You need to verify agent understanding
7. **Spreadsheet Integration**: You have trait data in Excel format
8. **Experimental Research**: You're testing new conversation mechanisms

### Migration Guide

**From Standard to Prototype:**

1. Update `Utterance` field: `speaker` → `actor`
2. Update `TurnLog` fields: `speaker/listener` → `actor/audience`
3. Add trait paragraph generation in `ConversationStudy.run()`
4. Enable self-reflection in `act_based_on_belief()` and `audience_evaluate_and_respond()`
5. Add question checks in conversation loop
6. Update prompt headers with world-building narrative
7. Change agent names and roles

**From Prototype to Standard:**

1. Remove trait paragraph generation
2. Remove self-reflection calls
3. Remove question checks
4. Simplify prompt headers
5. Update field names (`actor` → `speaker`)
6. Remove spreadsheet support
7. Simplify `TurnLog` structure

---

## Performance Comparison

### LLM Call Count Per Turn

**Standard:**
```
Turn 1:
  - Actor act: 1 call
  - Audience evaluate: 1 call
  - Audience respond: 1 call
  - Actor update (measurement): 1 call
  - Actor learning: 1 call
Total: 5 calls

Turn 2+:
  - Actor act_based_on_belief: 1 call
  - Audience evaluate: 1 call
  - Audience respond: 1 call
  - Actor update (measurement): 1 call
  - Actor learning: 1 call
Total: 5 calls per turn
```

**Prototype:**
```
Initialization:
  - Actor trait paragraph: 1 call
  - Audience trait paragraph: 1 call
Total: 2 calls

Turn 1:
  - Actor act: 1 call
  - Actor self-reflection (if traits): 1 call
  - Audience evaluate: 1 call
  - Audience respond: 1 call
  - Audience self-reflection (if norms/traits): 1 call
  - Actor update (measurement): 1 call
  - Actor learning: 1 call
  - Actor question check: 2 calls
  - Audience question check: 2 calls
Total: 11-13 calls

Turn 2+:
  - Actor act_based_on_belief: 1 call
  - Actor self-reflection (if traits): 1 call
  - Audience evaluate: 1 call
  - Audience respond: 1 call
  - Audience self-reflection (if norms/traits): 1 call
  - Actor update (measurement): 1 call
  - Actor learning: 1 call
  - Actor question check: 2 calls
  - Audience question check: 2 calls
Total: 11-13 calls per turn
```

**Cost Impact**: Prototype uses **2-2.6x more LLM calls**, significantly increasing API costs.

---

## Feature Matrix

| Feature | Standard | Prototype | Notes |
|---------|----------|-----------|-------|
| **Core PE System** | ✅ | ✅ | Both identical |
| **Particle Filter** | ✅ | ✅ | Both identical |
| **Cultural Norms** | ✅ Simple | ✅ World-building | Different approaches |
| **Personality Traits** | ✅ Scores | ✅ Paragraphs | Different methods |
| **Self-Reflection** | ❌ | ✅ | Prototype only |
| **Option Generation** | ❌ | ✅ | Prototype only |
| **Question Checks** | ❌ | ✅ | Prototype only |
| **Spreadsheet Support** | ❌ | ✅ | Prototype only |
| **Plotting** | ✅ Enabled | ⚠️ Disabled | Prototype has it but commented |
| **Debug Prints** | ❌ | ✅ | Prototype has debug code |
| **Initialization Traits** | ❌ | ✅ | Prototype generates paragraphs |

---

## Conclusion

Both implementations serve different purposes:

- **`pe_conversation_openai.py`**: Production-ready, clean, cost-effective standard implementation
- **`pe_conversation_prototype.py`**: Research-focused, feature-rich experimental implementation

Choose based on your specific needs:
- **Standard** for general use, cost efficiency, and maintainability
- **Prototype** for advanced research, stigma studies, and experimental features

---

## Appendix: Code Snippets Comparison

### Prompt Header Generation

**Standard:**
```python
def _prompt_header(self) -> str:
    norms_text = ""
    if self.cultural_norms:
        norms_text = "CULTURAL NORMS YOU FOLLOW:\n" + "\n".join(
            f"- {n.name}: {n.description}" for n in self.cultural_norms
        ) + "\n\n"

    traits_text = ""
    if self.traits:
        traits_text = "YOUR PERSONALITY TRAITS:\n" + "\n".join(
            f"- {t.name} ({self.trait_scores.get(t.name, 'NA')} / 3): {t.assertion}"
            for t in self.traits
        ) + "\n\n"

    return norms_text + traits_text
```

**Prototype:**
```python
def _prompt_header(self) -> str:
    norms_prompt = f"""You are initializing a social simulation of an
    alternate reality called 2A25. The dominant cognitive-cultural order
    is the Cadens majority who all follow these cultural norms: {norms_text}.
    [Extensive world-building narrative...]"""

    if self.trait_paragraph:
        traits_prompt = f"""The following paragraph describes {self.name}
        interactions and how they perceive, process, and interact with the
        social world.{self.trait_paragraph}"""

    return f"{norms_prompt}{context_prompt}You are {self.name}. {self.memory.goal.description}.{traits_prompt}"
```

### Trait Initialization

**Standard:**
```python
# Traits passed directly to agent, used in prompts as scores
agent = Agent(
    name="John",
    traits=traits,
    trait_scores={...}  # Scores provided
)
```

**Prototype:**
```python
# Traits converted to paragraphs during initialization
actor_traits_paragraph = actor.initialize_personality_traits(actor.traits)
actor.trait_paragraph = actor_traits_paragraph
# Paragraph used in prompts
```

---

**Document Version**: 1.0
**Last Updated**: 2025-12-27
**Files Compared**:
- `pe_conversation_openai.py` (standard)
- `pe_conversation_prototype.py` (prototype)
