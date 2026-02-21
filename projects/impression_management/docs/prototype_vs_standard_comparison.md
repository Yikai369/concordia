# Detailed Comparison: `impression_management_standard` vs `pe_conversation_prototype.py`

## Overview

This document provides a comprehensive comparison between two implementations of the Impression Management PE (Prediction Error) conversation system:

1. **`impression_management_standard`**: Production-ready, framework-integrated implementation using Concordia
2. **`pe_conversation_prototype.py`**: Experimental standalone implementation with advanced trait handling

---

## Table of Contents

1. [Architecture Comparison](#architecture-comparison)
2. [Feature Differences](#feature-differences)
3. [Data Structure Differences](#data-structure-differences)
4. [Personality Trait System](#personality-trait-system)
5. [Cultural Norms Implementation](#cultural-norms-implementation)
6. [Agent Configuration](#agent-configuration)
7. [Code Quality & Maintainability](#code-quality--maintainability)
8. [Output Formats](#output-formats)
9. [Use Cases & Recommendations](#use-cases--recommendations)

---

## Architecture Comparison

### Framework Integration

| Aspect | `impression_management_standard` | `pe_conversation_prototype.py` |
|--------|----------------------------------|--------------------------------|
| **Architecture** | Concordia framework | Standalone Python script |
| **Execution Model** | `sim.play()` automatic loop | Manual conversation loop |
| **Component System** | Component-based (IMPE components) | Direct `Agent` class methods |
| **Dependencies** | Requires Concordia framework | No framework dependencies |
| **Modularity** | Multiple files (config, setup, etc.) | Single file (~1185 lines) |

### Execution Flow

**Standard (Framework-based):**
```python
# Uses Concordia's standard simulation loop
sim = simulation.Simulation(config=config, model=model, embedder=embedder)
results_log = sim.play(max_steps=args.turns * 2, raw_log=raw_log)
turn_logs = extract_turn_data_from_entities(sim, agent_a_name, agent_b_name, args.turns)
```

**Prototype (Manual Loop):**
```python
# Manual 4-step conversation loop
for t in range(1, self.total_turns + 1):
    actor_utt = actor.act_based_on_belief(...)  # Step 1: ACT
    I_t, audience_reply = audience.audience_evaluate_and_respond(...)  # Step 2: OBSERVE
    I_hat, ess = actor.actor_update_particles(...)  # Step 3: UPDATE
    refl = actor.learning(turn=t)  # Step 4: LEARNING
```

---

## Feature Differences

### 1. Personality Trait System

#### Standard Implementation

**Trait Definition:**
```python
@dataclass
class PersonalityTrait:
    name: str
    assertion: str

ALL_TRAITS: List[PersonalityTrait] = [
    PersonalityTrait("Detail-focused", "I tend to focus on individual parts..."),
    PersonalityTrait("Avoids eye contact", "I do not make eye contact..."),
    # ... 11 total traits
]
```

**Trait Application:**
- Traits applied directly via `PersonalityTraitsComponent`
- Uses scoring system (0-3 scale)
- Audience: scores 2-3, Actor: scores 0-1
- Traits included in prompt headers directly

#### Prototype Implementation

**Trait Definition:**
```python
@dataclass
class PersonalityTrait:
    survey: str  # Different field name!
    assertion: str

# Traits loaded from Excel spreadsheet
def extract_traits_from_spreadsheet(file_path: str) -> List[PersonalityTrait]:
    df = pd.read_excel(file_path, header=0)
    traits = []
    for survey in df.columns:
        series = df[survey].dropna()
        for assertion in series.astype(str):
            traits.append(PersonalityTrait(survey=survey, assertion=assertion))
    return traits
```

**Trait Paragraph Generation:**
```python
def initialize_personality_traits(self, traits: List[PersonalityTrait]) -> str:
    """Generates a paragraph describing personality from traits"""
    intro = "Write a detailed paragraph describing this person..."
    trait_list = "\n".join(f"- {s.assertion}" for s in traits)
    prompt = f"{intro}\n{trait_list}"

    traits_paragraph = self.llm(prompt)  # LLM generates paragraph
    set_traits_prompt = f"You are {self.name}. The following paragraph..."
    self.llm(set_traits_prompt + traits_paragraph)

    return traits_paragraph
```

**Key Differences:**
- ✅ **Prototype**: Generates natural language paragraph from traits
- ✅ **Prototype**: Loads traits from Excel spreadsheet
- ✅ **Prototype**: Stores trait paragraphs in JSON output
- ❌ **Standard**: Uses fixed trait list
- ❌ **Standard**: No paragraph generation

### 2. Option Space Generation (Prototype Only)

**Experimental Feature:**
```python
def generate_option_space(self, prompt) -> List[Tuple[str, str]]:
    """Generate 4 distinct options for how to respond next"""
    options_prompt = """Generate a set of 4 distinct options...
        Output in this format exactly:
        <numerical option number>.
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
    """
    raw_output = self.llm(prompt + options_prompt)
    # Parse and return 4 options
    return options

def choose_option(self, options: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Choose one option with mental deliberation"""
    choice_prompt = """Choose one of the four options...
        Mentally deliberate on why you chose this option...
    """
    # LLM chooses and returns selected option
    return options[choice_idx]
```

**Status:** Currently commented out in prototype code (lines 618-622)

**Standard:** No option generation feature

### 3. Self-Reflection Methods

#### Prototype

**Detailed Consistency Checking:**
```python
def audience_self_reflection(self, actor_utt: Utterance, audience_reply: Utterance, I_t: float) -> str:
    """Assesses and improves response for consistency"""
    critique_prompt = f"""You just replied with:
        {audience_reply.text}  Body language: "{audience_reply.body}"

        Is it consistent with who you are and what you know about the world?
    """
    critique = self.llm(critique_prompt)
    return critique

def actor_self_reflection(self, actor_utt: Utterance, aud_utt: Utterance) -> str:
    """Checks actor response consistency with personality"""
    # Similar consistency check
```

**Usage:**
```python
# In act_based_on_belief() - lines 833-838
if self.traits:
    new_resp_raw = self.actor_self_reflection(utt, audience_last_utt)
    new_dlg, new_body = self.format_response(new_resp_raw)
    new_utt = Utterance(turn=turn, actor=self.name, text=new_dlg, body=new_body)
    final_utt = new_utt  # ⚠️ ALWAYS replaces original response
```

**Revision Behavior:**
- ✅ **Does revise**: Always replaces original response with self-reflection response if traits exist
- ❌ **No threshold check**: No consistency score - always replaces unconditionally
- ❌ **No feedback loop**: Doesn't check if revision is better, just replaces
- ⚠️ **Note**: `audience_self_reflection()` method exists but is **never called** (only actor revises)

#### Standard

**Threshold-Based Self-Assessment:**
- Uses `IMPESelfAssessmentComponent` (optional, via `--enable_self_assessment`)
- Checks consistency with norms and traits
- **Computes consistency score** (0-1 scale)
- **Only revises if** consistency < threshold (default: 0.7) AND `enable_revision=True`
- Provides feedback on what's inconsistent
- Can disable revision while still logging assessments (`--disable_revision`)

**Revision Behavior:**
```python
# Standard version checks consistency score
consistency_score = assess_consistency(response, traits, norms, goal)
if consistency_score < threshold and enable_revision:
    revised_response = generate_revision(original_response, feedback)
    return revised_response
else:
    return original_response  # Keep original if consistent enough
```

**Key Differences:**

| Aspect | Prototype | Standard |
|--------|-----------|----------|
| **Revision Trigger** | Always (if traits exist) | Only if consistency < threshold |
| **Consistency Score** | ❌ No | ✅ Yes (0-1 scale) |
| **Threshold** | ❌ No | ✅ Yes (default: 0.7) |
| **Feedback** | ❌ No | ✅ Yes (explains inconsistencies) |
| **Conditional Revision** | ❌ No | ✅ Yes (can disable) |
| **Actor Revision** | ✅ Yes | ✅ Yes |
| **Audience Revision** | ❌ No (method exists but unused) | ✅ Yes (if enabled) |

---

## Data Structure Differences

### Utterance Class

**Standard:**
```python
@dataclass
class Utterance:
    turn: int
    speaker: str  # Uses "speaker"
    text: str
    body: str = ""
```

**Prototype:**
```python
@dataclass
class Utterance:
    turn: int
    actor: str  # Uses "actor" instead of "speaker"
    text: str
    body: str = ""
```

### TurnLog Class

**Standard:**
```python
@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str  # Uses "speaker"
    listener: str  # Uses "listener"
    speaker_text: str
    speaker_body: str
    audience_I: float
    audience_text: str
    audience_body: str
    actor_I_hat: float
    actor_pe: float
    reflection_text: str
    ess: float
```

**Prototype:**
```python
@dataclass
class TurnLog:
    time: str
    turn: int
    actor: str  # Uses "actor"
    audience: str  # Uses "audience"
    actor_text: str
    actor_body: str
    audience_I: float
    audience_text: str
    audience_body: str
    actor_I_hat: float
    actor_pe: float
    # Note: No reflection_text field
    ess: float
```

### AgentMemory

**Both implementations have identical structure:**
```python
@dataclass
class AgentMemory:
    goal: Goal
    conversation: List[Utterance]
    pe_history: List[PERecord]
    reflections: List[ReflectionRecord]
    pf_particles: List[float]
    pf_weights: List[float]
    pf_history: List[Dict[str, float]]
```

---

## Personality Trait System

### Trait Loading

**Standard:**
- Fixed list in `constants.py`
- 11 predefined traits
- Traits selected via `generate_trait_scores()` or `generate_parametric_traits()`

**Prototype:**
- Loads from Excel spreadsheet: `"autism-measures-compilation.xlsx"`
- Dynamic trait extraction from multiple survey columns
- Each column becomes a `survey` field
- Each row becomes an `assertion`

### Trait Application

**Standard:**
```python
# Traits included directly in prompt
traits_text = "YOUR PERSONALITY TRAITS:\n" + "\n".join(
    f"- {t.name} ({score}/3): {t.assertion}"
    for t, score in zip(traits, scores)
)
```

**Prototype:**
```python
# Step 1: Generate paragraph from traits
traits_paragraph = self.llm("Write a detailed paragraph...")

# Step 2: Set personality via paragraph
self.llm(f"You are {self.name}. The following paragraph describes...")

# Step 3: Store paragraph for later use
self.trait_paragraph = traits_paragraph
```

### Trait Storage in Output

**Standard:**
- Traits not stored in JSON output
- Only turn logs saved

**Prototype:**
```python
def save_json(self, path: str) -> None:
    output = {
        "actor_traits": self.actor_traits_paragraph,  # ✅ Stored
        "audience_traits": self.audience_traits_paragraph,  # ✅ Stored
        "turns": [asdict(l) for l in self.log]
    }
    json.dump(output, f, ensure_ascii=False, indent=2)
```

---

## Cultural Norms Implementation

### Norm Definitions

**Both use identical 18 cultural norms:**
- Stated purpose first
- Announced topics
- Direct, literal language
- Hidden agendas
- Optional small talk
- Respect for passions
- Generous common ground
- Low coordination pressure
- Slow conversational pacing
- Open clarification
- Eye contact
- Comfortable silence & parallel play
- Negotiated personal space
- Integrity over politeness
- Minimal figurative speech
- Preference of traits in others
- Balanced reciprocity
- Brief by default

### World-Building Context

**Standard:**
```python
# Generic world-building context
norms_prompt = """You are in an alternative world in the year 3025...
    These cultural norms govern all etiquettes...
"""
```

**Prototype:**
```python
# More detailed 2A25 world description
norms_prompt = """You are initializing a social simulation of an alternate reality called 2A25.
    The dominant cognitive-cultural order is the Cadens majority...
    In 2A25, there is another social minority group called the Riffers...
    This setting is a fictional social world. This world is not an allegory...
"""
```

### Interview Context

**Standard:**
- Generic Product Manager interview
- Configurable role via `Goal.role`

**Prototype:**
- Customer Service Agent role
- Hardcoded interview questions for Caden (lines 426-439)
- Hardcoded experiences for Riffer (lines 440-452)

**Example (Prototype):**
```python
if self.name == "Caden":
    context_prompt += """This is an Interview question bank. You can ask:
        - Tell me about your customer service experience
        - What does a good support interaction look like to you?
        - Walk me through how you handle an angry customer.
        # ... 10 more questions
    """
elif self.name == "Riffer":
    context_prompt += """You can mention any or all of the below experiences:
        Experience 1: Managed high-volume frontline support...
        Experience 2: Resolved billing disputes...
        # ... 4 experiences total
    """
```

---

## Agent Configuration

### Default Agent Names

| Implementation | Actor Name | Audience Name |
|----------------|------------|---------------|
| **Standard** | "John" | "Jane" |
| **Prototype** | "Riffer" | "Caden" |

### Role Configuration

**Standard:**
```python
interview_role = """
    Role: Product Manager

    Responsibilities:
    - Define product vision and strategy
    - Work with engineering teams to deliver features
    - Analyze user data to inform product decisions
    - Communicate with stakeholders across the organization

    Evaluation Criteria:
    - Technical understanding of product development
    - Ability to prioritize features and manage trade-offs
    - Communication skills and stakeholder management
    - Data-driven decision making
"""
```

**Prototype:**
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

### Agent Initialization

**Standard:**
```python
A = Agent(
    name="John",
    goal=actor_goal,
    llm=llm,
    cultural_norms=[],  # Actor has no norms
    traits=actor_traits,
    context=not args.no_context
)

B = Agent(
    name="Jane",
    goal=audience_goal,
    llm=llm,
    cultural_norms=aud_norms,  # Audience has norms
    traits=aud_traits,
    context=not args.no_context
)
```

**Prototype:**
```python
A = Agent(  # interviewee
    name="Riffer",
    goal=actor_goal,
    llm=llm,
    cultural_norms=aud_norms,  # ⚠️ Actor ALSO has norms!
    traits=actor_traits,
    context=not args.no_context
)

B = Agent(  # interviewer
    name="Caden",
    goal=audience_goal,
    llm=llm,
    cultural_norms=aud_norms,  # Audience has norms
    traits=aud_traits,
    context=not args.no_context
)
```

**Key Difference:** Prototype gives cultural norms to BOTH agents, standard only to audience.

---

## Code Quality & Maintainability

### Debug Code

**Prototype:**
```python
# Line 623: Debug print in audience_evaluate_and_respond()
print(resp_prompt)

# Line 827: Debug print in act_based_on_belief()
print(prompt)

# Line 921-922: Debug prints of trait paragraphs
print(actor_traits_paragraph)
print(audience_traits_paragraph)
```

**Standard:**
- No debug print statements
- Clean production code

### Commented Code

**Prototype:**
```python
# Lines 281-293: Commented-out ALL_TRAITS definition
# ALL_TRAITS: List[PersonalityTrait] = [
#     PersonalityTrait("Detail-focused", ...),
#     ...
# ]

# Lines 618-622: Commented-out option generation
# options = self.generate_option_space(resp_prompt)
# choice = self.choose_option(options)

# Lines 864-867: Commented-out question check fields
# actor_personality_check: str
# actor_context_check: str
# audience_personality_check: str
# audience_context_check: str

# Line 1170: Commented-out plotting
# plot_learning_dynamics(runlog, save_dir=study.save_dir)
```

**Standard:**
- Minimal commented code
- Clean, production-ready

### Hardcoded Values

**Prototype:**
```python
# Line 1133: Hardcoded file path
aud_traits = extract_traits_from_spreadsheet("autism-measures-compilation.xlsx")

# Lines 425-452: Hardcoded interview questions/experiences
if self.name == "Caden":
    context_prompt += """This is an Interview question bank..."""
elif self.name == "Riffer":
    context_prompt += """You can mention any or all of the below experiences..."""
```

**Standard:**
- Configurable paths
- Generic, reusable code

### Error Handling

**Standard:**
- Framework-managed error handling
- Component-level error recovery

**Prototype:**
- Manual error handling
- Explicit retry logic with exponential backoff

---

## Output Formats

### JSON Structure

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
    "actor_pe": 0.05,
    "reflection_text": "...",
    "ess": 150.5
  }
]
```

**Prototype:**
```json
{
  "actor_traits": "A detailed paragraph describing Riffer's personality...",
  "audience_traits": "A detailed paragraph describing Caden's personality...",
  "turns": [
    {
      "time": "2025-12-27T23:04:37Z",
      "turn": 1,
      "actor": "Riffer",
      "audience": "Caden",
      "actor_text": "...",
      "actor_body": "...",
      "audience_I": 0.8,
      "audience_text": "...",
      "audience_body": "...",
      "actor_I_hat": 0.75,
      "actor_pe": 0.05,
      "ess": 150.5
    }
  ]
}
```

**Key Differences:**
- ✅ **Prototype**: Includes trait paragraphs at top level
- ✅ **Prototype**: Uses `actor`/`audience` field names
- ❌ **Prototype**: No `reflection_text` field
- ✅ **Standard**: Includes `reflection_text` field
- ✅ **Standard**: Uses `speaker`/`listener` field names

### Plot Generation

**Standard:**
- ✅ Plots enabled by default
- Can disable with `--no_plots` flag
- Generates: `pe.png`, `delta_I.png`, `learning_gain.png`

**Prototype:**
- ❌ Plotting commented out (line 1170)
- Same plotting function exists but not called

---

## Use Cases & Recommendations

### When to Use `impression_management_standard`

✅ **Use Standard When:**
- You need framework integration with Concordia
- You want production-ready, maintainable code
- You need configurable agent names and roles
- You want clean, modular architecture
- You need component-based extensibility
- You want standard Concordia patterns

**Best For:**
- Research projects requiring framework integration
- Production deployments
- Code that needs to be maintained long-term
- Projects requiring multiple agent configurations

### When to Use `pe_conversation_prototype.py`

✅ **Use Prototype When:**
- You need spreadsheet-based trait loading
- You want LLM-generated personality paragraphs
- You're experimenting with option space generation
- You need detailed self-reflection mechanisms
- You want trait paragraphs in output JSON
- You're doing exploratory research

**Best For:**
- Experimental research on personality modeling
- Studies requiring dynamic trait loading
- Prototyping new features
- One-off experiments
- Research on trait paragraph generation

### Migration Path

**From Prototype to Standard:**

1. **Trait System:**
   - Convert spreadsheet traits to fixed list in `constants.py`
   - Remove trait paragraph generation (or implement as optional component)
   - Update `PersonalityTrait` to use `name` instead of `survey`

2. **Data Structures:**
   - Change `Utterance.actor` → `Utterance.speaker`
   - Change `TurnLog.actor`/`audience` → `TurnLog.speaker`/`listener`
   - Add `reflection_text` to `TurnLog`

3. **Agent Configuration:**
   - Remove hardcoded interview questions
   - Make agent names configurable
   - Remove cultural norms from actor (if desired)

4. **Code Quality:**
   - Remove debug print statements
   - Remove commented code
   - Make file paths configurable

5. **Framework Integration:**
   - Convert `Agent` class to Concordia components
   - Replace manual loop with `sim.play()`
   - Use component lifecycle methods

---

## Feature Comparison Matrix

| Feature | Standard | Prototype | Notes |
|---------|----------|-----------|-------|
| **Framework Integration** | ✅ Yes | ❌ No | Standard uses Concordia |
| **Particle Filter** | ✅ Yes | ✅ Yes | Identical implementation |
| **Cultural Norms** | ✅ Yes | ✅ Yes | Same 18 norms |
| **Personality Traits** | ✅ Fixed list | ✅ Spreadsheet | Different sources |
| **Trait Paragraphs** | ❌ No | ✅ Yes | Prototype generates via LLM |
| **Option Generation** | ❌ No | ✅ Yes | Experimental in prototype |
| **Self-Reflection** | ⚠️ Optional | ✅ Always | Prototype more detailed |
| **Plotting** | ✅ Enabled | ❌ Disabled | Prototype commented out |
| **Debug Code** | ❌ No | ✅ Yes | Prototype has prints |
| **Hardcoded Values** | ❌ No | ✅ Yes | Prototype has file paths |
| **Configurable Names** | ✅ Yes | ❌ No | Prototype hardcoded |
| **Configurable Roles** | ✅ Yes | ❌ No | Prototype hardcoded |
| **JSON Output** | ✅ Turns only | ✅ Turns + traits | Prototype includes paragraphs |
| **Modular Structure** | ✅ Yes | ❌ No | Standard multi-file |
| **Production Ready** | ✅ Yes | ⚠️ Experimental | Standard is cleaner |

---

## Conclusion

**`impression_management_standard`** is the **production-ready, framework-integrated** implementation suitable for:
- Long-term research projects
- Framework-based simulations
- Maintainable, extensible code
- Standard Concordia patterns

**`pe_conversation_prototype.py`** is an **experimental standalone** implementation with:
- Advanced trait paragraph generation
- Spreadsheet-based trait loading
- Experimental option generation
- Detailed self-reflection mechanisms

**Recommendation:** Use `impression_management_standard` for production work, and reference `pe_conversation_prototype.py` for experimental features that could be ported to the standard version.

---

## Appendix: Code Snippets

### Trait Paragraph Generation (Prototype)

```python
def initialize_personality_traits(self, traits: List[PersonalityTrait]) -> str:
    """Set behaviour profile with personality traits for the agent."""
    if not self.traits:
        return None

    intro = "Write a detailed paragraph describing this person based on statements about them. Consider how they would perceive, process, and interact with the social world. The statements are as follows:"
    trait_list = "\n".join(f"- {s.assertion}" for s in traits)

    prompt = f"""{intro}
    {trait_list}
    """

    traits_paragraph = self.llm(prompt)

    set_traits_prompt = f"""You are {self.name}. The following paragraph describes {self.name} interactions and how they perceive, process, and interact with the social world."""
    self.llm(set_traits_prompt + traits_paragraph)

    return traits_paragraph
```

### Spreadsheet Trait Extraction (Prototype)

```python
def extract_traits_from_spreadsheet(file_path: str) -> List[PersonalityTrait]:
    """Extract personality traits for the specified agent from the spreadsheet."""
    df = pd.read_excel(file_path, header=0)
    traits: list[PersonalityTrait] = []

    for survey in df.columns:
        series = df[survey].dropna()

        for assertion in series.astype(str):
            assertion = assertion.strip()
            if assertion:
                traits.append(
                    PersonalityTrait(
                        survey=survey,
                        assertion=assertion
                    )
                )
    return traits
```

### Option Space Generation (Prototype)

```python
def generate_option_space(self, prompt) -> List[Tuple[str, str]]:
    """Generate a set of 4 distinct options for how to respond next."""
    options_prompt = """Generate a set of 4 distinct options for how to respond next in the conversation. Each option should include a brief reply and body language. Format the output as a numbered list:
        Output in this format exactly:
        <numerical option number>.
        DIALOGUE: <one sentence>
        BODY: <brief body language phrase>
    """

    raw_output = self.llm(prompt + options_prompt)

    options = []
    option_blocks = re.split(r'\n\s*[1-4]\s*[.)\s]', raw_output)
    option_blocks = option_blocks[1:5]

    for block in option_blocks:
        if block.strip():
            dlg, body = self.format_response(block)
            options.append((dlg, body))

    return options
```

---

**Document Version:** 1.0
**Last Updated:** 2025-12-27
**Author:** Auto-generated comparison documentation
