# PE Conversation System Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Core Concepts](#core-concepts)
4. [Data Structures](#data-structures)
5. [Key Classes](#key-classes)
6. [Workflow](#workflow)
7. [Configuration](#configuration)
8. [Usage Examples](#usage-examples)
9. [API Reference](#api-reference)
10. [Output and Analysis](#output-and-analysis)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The **PE (Prediction Error) Conversation System** is a framework for simulating adaptive conversations between two LLM-powered agents. The system implements a prediction error-driven learning mechanism where one agent (the **actor**) attempts to achieve a goal (e.g., being perceived as competent) while the other agent (the **audience/listener**) evaluates the actor's performance.

### Key Features

- **Particle Filter Belief Tracking**: The actor uses a particle filter to maintain beliefs about the audience's hidden evaluation state
- **Prediction Error Learning**: Agents adapt their behavior based on prediction errors between expected and actual outcomes
- **Cultural Norms & Personality Traits**: Configurable cultural norms and personality traits that shape agent behavior
- **Interview Context**: Optional interview simulation context for studying impression management
- **Comprehensive Logging**: Detailed per-turn logs with visualization support

### Use Cases

- Social simulation research
- Impression management studies
- Adaptive conversation modeling
- Cultural norm learning experiments
- Interview performance analysis

---

## System Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Conversation Study                        │
│                                                              │
│  ┌──────────┐         ┌──────────┐                         │
│  │  Actor   │◄───────►│ Audience │                         │
│  │  (John)  │         │  (Jane)  │                         │
│  └────┬─────┘         └────┬─────┘                         │
│       │                    │                                │
│       │ ACT                │ OBSERVE                        │
│       │                    │                                │
│       ▼                    ▼                                │
│  ┌──────────────────────────────────────┐                  │
│  │     Particle Filter Belief State     │                  │
│  │  (I_hat: estimated evaluation)       │                  │
│  └──────────────────────────────────────┘                  │
│       │                                                      │
│       │ LEARNING (Reflection)                                │
│       ▼                                                      │
│  ┌──────────────────────────────────────┐                  │
│  │      Turn Log & Visualization         │                  │
│  └──────────────────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Turn Structure

Each conversation turn follows this sequence:

1. **ACT**: Actor generates an utterance based on current belief
2. **OBSERVE**: Audience evaluates actor and responds
3. **UPDATE**: Actor updates particle filter with audience's response
4. **LEARN**: Actor reflects on performance and plans next action
5. **LOG**: System records all turn data

---

## Core Concepts

### Prediction Error (PE)

Prediction Error measures the difference between the actor's prior belief (`I_hat_prev`) and posterior belief (`I_hat`) about the audience's evaluation:

```
PE = |I_hat_prev - I_hat|
```

- **High PE**: Large belief update, indicating significant learning
- **Low PE**: Small belief update, indicating stable expectations

### Particle Filter

A Bayesian filtering technique that maintains a distribution over possible evaluation states:

- **Particles**: Sample points representing possible evaluation values [0,1]
- **Weights**: Probability mass assigned to each particle
- **Process Noise**: Random walk that diffuses particles over time
- **Observation Model**: Gaussian likelihood based on LLM-extracted measurements
- **Resampling**: Prevents particle degeneracy when weights become too concentrated

### Belief State (I_hat)

The actor's posterior mean estimate of the audience's evaluation:

```
I_hat = Σ(particle_i × weight_i)
```

This value ranges from 0 (not competent) to 1 (fully competent).

### True State (I_t)

The audience's actual hidden evaluation of the actor, extracted from the audience's response via LLM.

---

## Data Structures

### Goal

Represents an agent's objective.

```python
@dataclass
class Goal:
    name: str              # e.g., "competence"
    description: str       # Detailed goal description
    role: str             # Interview role context (optional)
    ideal: float = 1.0    # Target value (0-1)
```

**Example:**
```python
actor_goal = Goal(
    name="competence",
    description="Be perceived as highly competent in an interview",
    role="Product Manager",
    ideal=1.0
)
```

### Utterance

Represents a single conversational turn.

```python
@dataclass
class Utterance:
    turn: int        # Turn number
    speaker: str     # Agent name
    text: str        # Dialogue text
    body: str = ""   # Body language description
```

### PERecord

Stores prediction error information for a turn.

```python
@dataclass
class PERecord:
    turn: int
    partner_text: str    # Partner's utterance text
    estimate: float      # I_hat value
    pe: float           # Prediction error
```

### ReflectionRecord

Stores agent reflection/learning output.

```python
@dataclass
class ReflectionRecord:
    turn: int
    text: str    # Reflection text
```

### AgentMemory

Maintains all agent state and history.

```python
@dataclass
class AgentMemory:
    goal: Goal
    conversation: List[Utterance]
    pe_history: List[PERecord]
    reflections: List[ReflectionRecord]

    # Particle filter state
    pf_particles: List[float]      # Current particles
    pf_weights: List[float]        # Current weights
    pf_history: List[Dict]         # PF metadata per turn
```

### TurnLog

Complete record of a conversation turn.

```python
@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str
    listener: str
    speaker_text: str
    speaker_body: str
    audience_I: float              # True evaluation I_t
    audience_text0: str            # Initial audience response
    audience_body0: str
    audience_text: str             # Final audience response (after reflection)
    audience_body: str
    actor_I_hat: float            # Actor's belief
    actor_pe: float               # Prediction error
    actor_personality_check: str
    actor_context_check: str
    audience_personality_check: str
    audience_context_check: str
    ess: float                     # Effective sample size
```

### CulturalNorm

Defines a cultural rule that agents must follow.

```python
@dataclass
class CulturalNorm:
    name: str
    description: str
```

**Predefined Norms:**
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

### PersonalityTrait

Defines a personality characteristic.

```python
@dataclass
class PersonalityTrait:
    name: str
    assertion: str                    # Positive assertion
    negative_assertion: Optional[str]  # Negative assertion
```

**Predefined Traits:**
- Detail-focused
- Avoids eye contact
- Not laid back
- Dislikes spontaneity
- Repeats phrases
- Poor imagination
- Not social
- Takes things literally
- Number-interested
- Dislikes crowds
- Doesn't share enjoyment

---

## Key Classes

### ParticleFilter

Implements a 1-D particle filter for tracking scalar states in [0,1].

#### Initialization

```python
pf = ParticleFilter(
    num_particles=200,      # Number of particles
    process_sigma=0.03,    # Process noise std dev
    obs_sigma=0.08,         # Observation noise std dev
    rng=random.Random(42)   # Random number generator
)
```

#### Methods

**`initialize(particles=None)`**
- Initializes particles and uniform weights
- If `particles` provided, uses them directly
- Otherwise samples Gaussian around 0.5
- Returns: `(particles, weights)`

**`predict(particles)`**
- Applies Gaussian process noise (random walk)
- Clamps to [0,1] range
- Returns: predicted particles

**`update(particles, observation)`**
- Computes Gaussian likelihood weights
- Normalizes weights
- Computes ESS (Effective Sample Size)
- Resamples if ESS < 50% of particle count
- Returns: `(updated_particles, weights, ess, resampled)`

**`_systematic_resample(weights)`**
- Performs systematic resampling
- Returns: list of particle indices

### Agent

The main agent class that handles conversation, belief tracking, and learning.

#### Initialization

```python
agent = Agent(
    name="John",
    goal=actor_goal,
    llm=llm_function,
    recent_k=3,                    # Conversation window size
    seed=42,
    cultural_norms=[...],          # Optional
    traits=[...],                  # Optional
    trait_scores={...},            # Optional
    context=True                   # Interview context flag
)
```

#### Key Methods

**`act(turn: int) -> Utterance`**
- Generates initial utterance (turn 1)
- Uses goal and ideal value
- Returns formatted utterance with dialogue and body language

**`act_based_on_belief(turn: int, belief: float, audience_last_utt: Utterance) -> Utterance`**
- Generates utterance based on current belief `I_hat`
- Includes recent conversation history
- Includes recent `I_hat` history
- Optionally performs self-reflection if traits enabled
- Returns final utterance

**`audience_evaluate_and_respond(turn: int, actor_utt: Utterance) -> Tuple[float, Utterance, Utterance]`**
- Evaluates actor's performance → `I_t` (true evaluation)
- Generates initial response
- Optionally performs self-reflection for norm/trait alignment
- Returns: `(I_t, initial_utterance, final_utterance)`

**`actor_update_particles(turn: int, listener_utt: Utterance, pf_model=None) -> Tuple[float, float]`**
- Updates particle filter with listener's response
- Extracts measurement via LLM
- Computes posterior belief `I_hat`
- Updates PE history
- Returns: `(I_hat, ess)`

**`learning(turn: int) -> ReflectionRecord`**
- Generates reflection based on current `I_hat`
- Plans next-turn improvements
- Returns reflection record

**`_prompt_header() -> str`**
- Generates prompt header with:
  - Cultural norms (if enabled)
  - Interview context (if enabled)
  - Goal description
- Used in all LLM prompts

**`initialize_personality_traits(traits: List[str]) -> None`**
- Initializes agent with personality trait profile
- Sends behavioral profile prompt to LLM

**`question_check() -> Tuple[str, str]`**
- Checks agent's understanding of:
  - Personality traits
  - Conversation context
- Returns: `(personality_response, context_response)`

### ConversationStudy

Orchestrates the conversation and logging.

#### Initialization

```python
study = ConversationStudy(
    agent_a=actor_agent,
    agent_b=audience_agent,
    save_dir="./output",      # Optional, defaults to timestamped dir
    total_turns=6,
    seed=42
)
```

#### Methods

**`run() -> List[TurnLog]`**
- Executes conversation loop
- Returns list of turn logs

**`save_json(path: str) -> None`**
- Saves turn logs to JSON file

---

## Workflow

### Complete Turn Sequence

```python
# 1. Actor acts
if turn == 1:
    speaker_utt = speaker.act(turn=t)
else:
    I_hat_prev = speaker.memory.pf_history[-1]["I_hat"]
    speaker_utt = speaker.act_based_on_belief(
        turn=t,
        belief=I_hat_prev,
        audience_last_utt=last_listener_utt
    )

# 2. Audience evaluates and responds
I_t, listener_init_reply, listener_final_reply = \
    listener.audience_evaluate_and_respond(turn=t, actor_utt=speaker_utt)

# 3. Actor updates belief
I_hat, ess = speaker.actor_update_particles(
    turn=t,
    listener_utt=listener_final_reply
)

# 4. Actor learns
reflection = speaker.learning(turn=t)

# 5. Compute prediction error
if len(speaker.memory.pf_history) > 1:
    prev_I_hat = speaker.memory.pf_history[-2]["I_hat"]
    actor_pe = abs(prev_I_hat - I_hat)
else:
    actor_pe = 1.0

# 6. Log turn
turn_log = TurnLog(...)
```

### Particle Filter Update Process

```
1. Load current particles and weights
2. Predict: particles_pred = predict(particles)
3. Extract measurement from listener's response via LLM
4. Update weights: w_i = exp(-0.5 * ((meas - x_i) / obs_sigma)^2)
5. Normalize weights
6. Compute ESS = 1 / Σ(w_i^2)
7. If ESS < 0.5 * N: resample particles
8. Compute I_hat = Σ(particle_i × weight_i)
9. Store PF state and update PE history
```

---

## Configuration

### Command-Line Arguments

```bash
python pe_conversation_openai.py [OPTIONS]
```

**Options:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--turns` | int | 2 | Total conversation turns |
| `--model` | str | "gpt-4o-mini" | OpenAI model name |
| `--temperature` | float | 0.2 | Sampling temperature |
| `--top_p` | float | 0.9 | Top-p nucleus sampling |
| `--window` | int | 3 | Recent K turns for context |
| `--outfile` | str | "pe_conversation_log.json" | Output JSON filename |
| `--no_audience_norms` | flag | False | Disable cultural norms for audience |
| `--no_traits` | flag | False | Disable personality traits |
| `--no_context` | flag | False | Disable interview context |
| `--seed` | int | 7 | Random seed |
| `--save_dir` | str | None | Output directory (auto-generated if None) |

### Environment Variables

**Required:**
```bash
export OPENAI_API_KEY="sk-..."
```

**PowerShell:**
```powershell
$Env:OPENAI_API_KEY = "sk-..."
```

### LLM Configuration

The system supports two LLM backends:

**OpenAI (default):**
```python
llm = make_openai_llm(
    model="gpt-4o-mini",
    temperature=0.2,
    top_p=0.9,
    max_retries=3,
    timeout_s=30.0
)
```

**Local Llama (commented out):**
```python
llm = make_local_llm(model="llama3.1:8b")
```

---

## Usage Examples

### Basic Usage

```bash
# Simple 6-turn conversation
python pe_conversation_openai.py --turns 6

# With custom model and temperature
python pe_conversation_openai.py --turns 10 --model gpt-4 --temperature 0.5

# Without cultural norms
python pe_conversation_openai.py --turns 6 --no_audience_norms

# Without personality traits
python pe_conversation_openai.py --turns 6 --no_traits

# Without interview context
python pe_conversation_openai.py --turns 6 --no_context

# Custom output directory
python pe_conversation_openai.py --turns 6 --save_dir ./my_experiment
```

### Programmatic Usage

```python
from pe_conversation_openai import (
    Agent, Goal, ConversationStudy,
    make_openai_llm, ALL_CULTURAL_NORMS, ALL_TRAITS,
    generate_parametric_traits
)

# Setup
llm = make_openai_llm(model="gpt-4o-mini", temperature=0.2)

# Define goals
actor_goal = Goal(
    name="competence",
    description="Be perceived as highly competent",
    role="Product Manager",
    ideal=1.0
)

audience_goal = Goal(
    name="evaluate_competence",
    description="Evaluate interviewee's competence",
    role="Product Manager",
    ideal=None
)

# Generate traits
traits = ALL_TRAITS.copy()
aud_traits = generate_parametric_traits(traits, is_audience=True)
actor_traits = generate_parametric_traits(traits, is_audience=False)

# Create agents
actor = Agent(
    name="John",
    goal=actor_goal,
    llm=llm,
    cultural_norms=[],
    traits=actor_traits,
    context=True
)

audience = Agent(
    name="Jane",
    goal=audience_goal,
    llm=llm,
    cultural_norms=ALL_CULTURAL_NORMS,
    traits=aud_traits,
    context=True
)

# Run study
study = ConversationStudy(actor, audience, total_turns=6)
logs = study.run()

# Save results
study.save_json("./results.json")
```

### Custom Cultural Norms

```python
from pe_conversation_openai import CulturalNorm

custom_norms = [
    CulturalNorm(
        name="Custom Norm 1",
        description="Description of the norm..."
    ),
    CulturalNorm(
        name="Custom Norm 2",
        description="Another norm description..."
    )
]

agent = Agent(
    name="Agent",
    goal=goal,
    llm=llm,
    cultural_norms=custom_norms
)
```

### Custom Personality Traits

```python
from pe_conversation_openai import PersonalityTrait

custom_traits = [
    PersonalityTrait(
        name="Trait Name",
        assertion="I have this trait...",
        negative_assertion="I do not have this trait..."
    )
]

agent = Agent(
    name="Agent",
    goal=goal,
    llm=llm,
    traits=custom_traits
)
```

---

## API Reference

### LLM Functions

#### `make_openai_llm(model, temperature, top_p, max_retries, timeout_s) -> LLMFn`

Creates an OpenAI LLM callable.

**Parameters:**
- `model` (str): Model name (e.g., "gpt-4o-mini")
- `temperature` (float): Sampling temperature [0,2]
- `top_p` (float): Nucleus sampling parameter [0,1]
- `max_retries` (int): Maximum retry attempts
- `timeout_s` (float): Request timeout in seconds

**Returns:** `Callable[[str], str]` - LLM function

#### `make_local_llm(model) -> LLMFn`

Creates a local Llama LLM callable (requires Ollama).

**Parameters:**
- `model` (str): Model name (e.g., "llama3.1:8b")

**Returns:** `Callable[[str], str]` - LLM function

### Utility Functions

#### `generate_trait_scores(rng, trait_list, is_audience) -> Dict[str, int]`

Generates trait scores for agents.

**Parameters:**
- `rng` (Random): Random number generator
- `trait_list` (List[PersonalityTrait]): List of traits
- `is_audience` (bool): True for audience (scores 2-3), False for actor (scores 0-1)

**Returns:** Dictionary mapping trait names to scores

#### `generate_parametric_traits(trait_list, is_audience) -> List[str]`

Generates parametric trait assertions.

**Parameters:**
- `trait_list` (List[PersonalityTrait]): List of traits
- `is_audience` (bool): Use assertion (True) or negative_assertion (False)

**Returns:** List of trait assertion strings

#### `parse_index_list(s) -> List[int]`

Parses comma-separated 1-based indices.

**Parameters:**
- `s` (str): Comma-separated index string (e.g., "1,3,5")

**Returns:** List of 0-based indices

#### `select_by_indices(full_list, indices) -> List[Any]`

Selects items from a list by indices.

**Parameters:**
- `full_list` (List[Any]): Source list
- `indices` (List[int]): Indices to select

**Returns:** Selected items

### Visualization

#### `plot_learning_dynamics(log, save_dir) -> None`

Generates three plots:
1. Prediction Error across turns
2. True I_t vs Estimated I_hat
3. Learning Gain (|delta I_hat| / |PE|)

**Parameters:**
- `log` (List[TurnLog]): Turn log data
- `save_dir` (str): Directory to save plots

**Output Files:**
- `pe.png`: Prediction error plot
- `delta_I.png`: Belief tracking plot
- `learning_gain.png`: Learning gain plot

---

## Output and Analysis

### JSON Log Structure

```json
[
  {
    "time": "2025-12-27T23:04:37Z",
    "turn": 1,
    "speaker": "John",
    "listener": "Jane",
    "speaker_text": "I have successfully prioritized features...",
    "speaker_body": "Maintain confident posture...",
    "audience_I": 0.8,
    "audience_text0": "The interviewee displayed moderate competence...",
    "audience_body0": "Sitting still...",
    "audience_text": "The interviewee showed notable competence...",
    "audience_body": "Avoided eye contact...",
    "actor_I_hat": 0.75,
    "actor_pe": 0.15,
    "actor_personality_check": "...",
    "actor_context_check": "...",
    "audience_personality_check": "...",
    "audience_context_check": "...",
    "ess": 150.5
  },
  ...
]
```

### Output Directory Structure

```
temp/
└── 2025-12-27_23-04-37/
    ├── pe_conversation_log.json
    ├── pe.png
    ├── delta_I.png
    └── learning_gain.png
```

### Key Metrics

**Prediction Error (PE):**
- Measures belief update magnitude
- High PE = significant learning event
- Low PE = stable expectations

**Learning Gain:**
- Ratio of belief change to prediction error
- `gain = |delta I_hat| / |PE|`
- High gain = efficient learning

**Effective Sample Size (ESS):**
- Measures particle filter health
- Low ESS = particle degeneracy (needs resampling)
- High ESS = good particle diversity

**Belief Accuracy:**
- Compare `I_hat` (estimated) vs `I_t` (true)
- Tracks how well actor tracks audience's evaluation

---

## Troubleshooting

### Common Issues

**1. API Key Not Set**
```
ERROR: OPENAI_API_KEY is not set
```
**Solution:** Export the API key before running:
```bash
export OPENAI_API_KEY="sk-..."
```

**2. Import Errors**
```
ERROR: Failed to import OpenAI client
```
**Solution:** Install the OpenAI package:
```bash
pip install -U openai
```

**3. Timeout Errors**
**Solution:** Increase timeout or check network connection:
```python
llm = make_openai_llm(timeout_s=60.0)
```

**4. Particle Filter Degeneracy**
**Symptoms:** ESS consistently low, poor belief tracking
**Solution:**
- Increase `num_particles`
- Adjust `process_sigma` and `obs_sigma`
- Check measurement extraction quality

**5. LLM Format Errors**
**Symptoms:** Missing DIALOGUE/BODY in responses
**Solution:**
- The code includes fallback parsing
- Check prompt clarity
- Consider adjusting temperature

### Debugging Tips

**1. Enable Debug Output**
Add print statements in key methods:
```python
def actor_update_particles(...):
    print(f"Measurement: {meas}")
    print(f"I_hat: {I_hat}")
    print(f"ESS: {ess}")
```

**2. Check Prompt Quality**
Inspect generated prompts:
```python
prompt = agent._prompt_header()
print(prompt)
```

**3. Verify Particle Filter State**
```python
print(f"Particles: {agent.memory.pf_particles[:10]}")
print(f"Weights: {agent.memory.pf_weights[:10]}")
print(f"ESS: {ess}")
```

**4. Monitor Conversation Flow**
The code already prints turn progress:
```
--- Turn 1 ---
Actor speaks...
Listener evaluates and responds...
Actor updates belief...
Computing prediction error...
```

---

## Advanced Topics

### Custom Particle Filter Parameters

```python
pf = ParticleFilter(
    num_particles=500,      # More particles = smoother estimates
    process_sigma=0.05,     # Higher = more belief drift
    obs_sigma=0.1,          # Higher = less trust in observations
    rng=random.Random(42)
)

agent.actor_update_particles(turn=t, listener_utt=utt, pf_model=pf)
```

### Custom Goal Definitions

```python
custom_goal = Goal(
    name="persuasiveness",
    description="Be perceived as persuasive and convincing",
    role="Sales Representative",
    ideal=1.0
)
```

### Multi-Agent Extensions

The current system supports two agents, but can be extended:

```python
# Create multiple audience agents
audiences = [
    Agent(name=f"Judge{i}", goal=audience_goal, llm=llm)
    for i in range(3)
]

# Aggregate evaluations
I_t_aggregate = np.mean([a.evaluate(actor_utt) for a in audiences])
```

### Custom Evaluation Metrics

Add to `TurnLog`:
```python
@dataclass
class TurnLog:
    # ... existing fields ...
    custom_metric: float = 0.0
```

---

## References

### Key Papers/Concepts

- **Particle Filtering**: Bayesian state estimation for non-linear systems
- **Prediction Error Learning**: Error-driven adaptation mechanisms
- **Impression Management**: Social psychology of self-presentation
- **Cultural Norms**: Social rules and expectations

### Related Files

- `pe_conversation_prototype.py`: Alternative implementation with different features
- `information_flow_history_*.json`: Example log files
- `temp/`: Output directory for runs

---

## License and Credits

This system is designed for research purposes. Ensure compliance with:
- OpenAI API usage terms
- Data privacy regulations
- Institutional review board requirements (if applicable)

---

## Version History

- **Current Version**: Based on `pe_conversation_openai (3).py`
- **Features**: Particle filter, cultural norms, personality traits, interview context
- **Last Updated**: 2025-12-27

---

## Contact and Support

For questions or issues:
1. Check this documentation
2. Review code comments
3. Examine example outputs in `temp/` directory
4. Consult related research papers

---

## Appendix: Complete Example Run

```bash
$ export OPENAI_API_KEY="sk-..."
$ python pe_conversation_openai.py --turns 6 --model gpt-4o-mini

Building LLM...
Using interview context for both agents...
Audience cultural norms: ['Stated purpose first', 'Announced topics', ...]
Personality traits: ['Detail-focused', 'Avoids eye contact', ...]
--- Turn 1 ---
Actor speaks...
Listener evaluates and responds...
Actor updates belief...
Computing prediction error...
--- Turn 2 ---
...
Saved detailed log → temp/2025-12-27_23-04-37/pe_conversation_log.json
```

The system generates:
- JSON log with all turn data
- Three visualization plots
- Console output with conversation summary

