# Impression Management PE Conversation: Concordia Framework Implementation Plan

## Overview

This document outlines the implementation plan for converting the impression management PE conversation system (with particle filters, cultural norms, personality traits, and interview context) into the Concordia framework. The system implements a sophisticated two-agent conversation where an actor (interviewee) adapts their behavior based on prediction errors computed via a particle filter, while an audience (interviewer) evaluates performance and provides feedback.

**Note**: See `plans/impression_management_pe_conversation_plan_duplication_analysis.md` for a detailed analysis of existing Concordia components and what can be reused vs. created new.

## Original System Analysis

### Key Features:
1. **Particle Filter Belief Tracking**: Actor maintains a particle filter to track beliefs about audience's hidden evaluation state I_t
2. **Asymmetric Actor-Audience Roles**: Actor (interviewee) vs Audience (interviewer) with different goals and behaviors
3. **Cultural Norms**: Configurable cultural norms that influence audience behavior
4. **Personality Traits**: Personality trait scoring system (0-3 scale) for both agents
5. **Interview Context**: Optional interview scenario with role-specific goals
6. **Body Language**: Utterances include both dialogue text and body language descriptions
7. **True Hidden State**: Audience generates true evaluation I_t ∈ [0,1] that actor must infer
8. **ESS Tracking**: Effective Sample Size tracking for particle filter diagnostics
9. **Learning Dynamics**: Reflection generation based on belief state
10. **Visualization**: Plotting of PE, I_t/I_hat trajectories, and learning gain

### Turn Flow:
1. **Actor acts**: Generates utterance (dialogue + body language) based on current belief I_hat
2. **Audience evaluates**: Produces true hidden evaluation I_t and feedback response
3. **Actor updates particles**: Updates particle filter belief given audience's response, computes new I_hat
4. **Actor reflects**: Generates reflection based on current belief state
5. **Log turn**: Record all turn data including I_t, I_hat, PE, ESS, reflections

### Data Structures:
- `Goal`: name, description, role (interview context), ideal
  - **Note**: Extends existing `Goal` from `pe_conversation.py` by adding `role: str | None = None` field
- `Utterance`: turn, speaker, text, body (body language)
  - **Note**: Extends existing `Utterance` from `pe_conversation.py` by adding `body: str = ""` field
- `PERecord`: turn, partner_text, estimate (I_hat), pe (signed: previous I_hat - current I_hat)
  - **Note**: Reuses existing `PERecord` from `pe_conversation.py` as-is
- `ReflectionRecord`: turn, text
  - **Note**: Reuses existing `ReflectionRecord` from `pe_conversation.py` as-is
- `EvaluationRecord`: turn, I_t (true hidden state), utterance (NEW - for storing audience evaluations)
  - **Note**: New data class, no existing equivalent
- `AgentMemory`: goal, conversation, pe_history, reflections, pf_particles, pf_weights, pf_history, evaluation_history (NEW)
- `TurnLog`: time, turn, speaker, listener, speaker_text, speaker_body, audience_I (I_t), audience_text, audience_body, actor_I_hat, actor_pe (absolute value), reflection_text, ess
- `CulturalNorm`: name, description
- `PersonalityTrait`: name, assertion
- `ParticleFilter`: 1D particle filter for [0,1] state space with Gaussian process/observation models

## Concordia Implementation Plan

### 0. Reusing Existing Components

Concordia already has a PE conversation system (`concordia/components/agent/pe_conversation.py`). We will:

**Reuse/Extend:**
- ✅ `Goal` - Extend with optional `role: str | None = None` field (for interview context)
- ✅ `Utterance` - Extend with optional `body: str = ""` field (for body language)
- ✅ `PERecord` - Reuse as-is (same structure)
- ✅ `ReflectionRecord` - Reuse as-is (same structure)
- ✅ `PEMemoryComponent` - Extend for `IMPEMemoryComponent` (add particle filter state, evaluation_history)

**Create New:**
- ❌ `EvaluationRecord` - New data class (for storing I_t in audience memory)
- ❌ `CulturalNorm` - New data class
- ❌ `PersonalityTrait` - New data class
- ❌ `ParticleFilter` - New utility class (no existing equivalent)
- ❌ All IMPE components - New (different algorithms: particle filter vs simple estimation)
- ❌ All prefabs - New (different component set)

**Rationale:**
- Existing PE system uses simple LLM estimation (PE = ideal - estimate)
- Impression management uses particle filter for belief tracking (different algorithm)
- Audience evaluation includes cultural norms/traits (different logic)
- Reflection based on I_hat (belief) not PE value (different logic)
- Act component handles first turn differently and includes body language (different logic)

**Import Strategy:**
```python
# In impression_management_pe.py
from concordia.components.agent.pe_conversation import (
    Goal,  # Extended with role field
    Utterance,  # Extended with body field
    PERecord,  # Reused as-is
    ReflectionRecord,  # Reused as-is
    PEMemoryComponent,  # Base class for IMPEMemoryComponent
)
```

### 1. Custom Components

All components will be placed in `concordia/components/agent/impression_management_pe.py` to maintain separation from the simpler PE conversation implementation.

#### 1.1 IMPEMemoryComponent
- **Purpose**: Store conversation history, PE records, reflections, goal, particle filter state, and evaluation records
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Inheritance**: Extends `PEMemoryComponent` from `pe_conversation.py` to reuse conversation/PE/reflection storage logic
- **State**:
  - `conversation`: List of `Utterance` objects (turn, speaker, text, body)
  - `pe_history`: List of `PERecord` objects (turn, partner_text, estimate, pe) - pe is signed
  - `reflections`: List of `ReflectionRecord` objects (turn, text)
  - `evaluation_history`: List of `EvaluationRecord` objects (turn, I_t, utterance) - NEW for storing audience evaluations
  - `goal`: `Goal` object (name, description, role, ideal)
  - `recent_k`: Window size for recent history (default 3)
  - `pf_particles`: List[float] - current particle filter particles
  - `pf_weights`: List[float] - current particle filter weights
  - `pf_history`: List[Dict[str, Any]] - particle filter history (turn, prior_mean, I_hat, ess, resampled, measurement)
- **Methods**:
  - `add_utterance(turn, speaker, text, body)`: Add conversation entry with body language
  - `add_pe_record(turn, partner_text, estimate, pe)`: Add PE record (pe is signed: prev_I_hat - current_I_hat)
  - `add_reflection(turn, text)`: Add reflection
  - `add_evaluation_record(turn, I_t, utterance)`: Add evaluation record (NEW - for audience to store I_t)
  - `update_particle_filter_state(particles, weights, history_entry)`: Update PF state
  - `get_recent_conversation(k)`: Get last k conversation entries (returns empty list if k is None or no conversation)
  - `get_recent_pe_history(k)`: Get last k PE records
  - `get_recent_reflections(k)`: Get last k reflections
  - `get_recent_evaluations(k)`: Get last k evaluation records (NEW)
  - `get_pf_history(k)`: Get last k PF history entries
  - `format_conversation(utterances)`: Format list of utterances as string (NEW - for prompt inclusion)
    - Format: `"- [t={turn} {speaker}] {text}"` per utterance
    - Returns `"- (none)"` if empty list
  - `get_state()` / `set_state()`: For checkpointing (must serialize PF state, all lists)
- **Key Design Decisions**:
  - Store PF state directly in memory component for easy access
  - PF history includes all diagnostic information (ESS, resampling flags, measurements)
  - Support body language in utterances for multimodal communication
  - Evaluation history stores I_t for audience (enables data extraction)
  - All data structures must be JSON-serializable for checkpointing

#### 1.2 ParticleFilter Utility Class
- **Purpose**: 1D particle filter implementation for belief tracking
- **Type**: Standalone utility class (not a component)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Methods**:
  - `__init__(num_particles, process_sigma, obs_sigma, rng)`: Initialize filter
  - `initialize(particles=None)`: Initialize particles and weights
  - `predict(particles)`: Apply process noise (Gaussian random walk)
  - `update(particles, observation)`: Weight particles by observation likelihood, resample if needed
  - `_systematic_resample(weights)`: Systematic resampling algorithm
- **Parameters**:
  - `num_particles`: Default 200
  - `process_sigma`: Default 0.03 (process noise)
  - `obs_sigma`: Default 0.08 (observation noise)
- **Key Design Decisions**:
  - Keep as utility class, not component (used by components)
  - ESS threshold: 0.5 * num_particles for resampling
  - Clamp particles to [0,1] range

#### 1.3 IMPEAudienceEvaluationComponent
- **Purpose**: Audience (interviewer) evaluates actor's performance and generates true hidden state I_t
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Dependencies**: IMPEMemoryComponent, LanguageModel, CulturalNormsComponent (optional), PersonalityTraitsComponent (optional)
- **Methods**:
  - `pre_observe(observation)`: Extract actor's utterance (text + body) from observation
    - Parse observation format: `"Actor said: \"{text}\"\nBody language: \"{body}\""`
    - Extract text and body via regex or string parsing
  - `post_observe()`:
    1. Call LLM to evaluate actor's performance → I_t ∈ [0,1]
    2. Generate feedback response (dialogue + body language) matching I_t
    3. Store I_t and response utterance in memory via `add_evaluation_record()`
- **Logic**:
  - Parse observation to extract actor's utterance (text and body language)
  - Determine actor name: "interviewee" if context=True, "partner" if context=False
  - Build evaluation prompt:
    - Include `_prompt_header()` (norms + traits if available)
    - Include goal description and role (if context enabled)
    - Include actor's utterance and body language
    - Prompt: "Rate {actor_name}'s competence on [0,1] scale. Options: [0.0, 0.1, ..., 1.0]"
  - Extract numeric rating via regex: `r"([01](?:\.\d+)?)"`
    - Fallback to 0.5 if no match
    - Clamp to [0,1] range
  - Build response prompt:
    - Include `_prompt_header()`
    - Include goal and role context
    - Include I_t score: "You rated {actor_name} with score {I_t:.2f}"
    - Include recent conversation history (last recent_k turns)
    - Prompt: "Produce a short reply that reflects your evaluation and matches your score"
    - Format: "DIALOGUE: <text>\nBODY: <body language>"
  - Parse response:
    - Extract dialogue: `r"DIALOGUE:\s*(.*)"` - fallback to entire response if not found
    - Extract body: `r"BODY:\s*(.*)"` - fallback to empty string if not found
  - Create Utterance and store in conversation
  - Store I_t and utterance via `memory.add_evaluation_record(turn, I_t, utterance)`
- **Cultural Norms & Traits Integration**:
  - Include cultural norms in evaluation prompt (if audience has norms)
  - Include personality traits with scores in prompt (format: "Trait Name (score/3): assertion")
  - Adjust evaluation criteria based on interview context (if enabled)
- **Key Design Decisions**:
  - True hidden state I_t is generated here (not inferred)
  - Response must match sentiment of I_t score
  - Body language generation is part of response
  - I_t stored in evaluation_history for data extraction
  - Context determines actor name (interviewee vs partner)

#### 1.4 IMPEActorParticleFilterComponent
- **Purpose**: Actor updates particle filter belief based on audience's response
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Dependencies**: IMPEMemoryComponent, LanguageModel, ParticleFilter
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Methods**:
  - `pre_observe(observation)`: Extract audience's response from observation
    - Parse observation format: `"Audience said: \"{text}\"\nBody language: \"{body}\""`
  - `post_observe()`:
    1. Initialize PF if first turn (particles and weights empty)
    2. Predict step: diffuse particles with process noise
    3. Use LLM to extract measurement from audience's response
    4. Update step: weight particles by measurement likelihood (Gaussian)
    5. Compute ESS, resample if ESS < 0.5 * num_particles
    6. Compute posterior mean I_hat
    7. Compute PE = previous I_hat - current I_hat (signed)
    8. Store PF state and PE record in memory
- **Logic**:
  - Load or initialize PF state:
    - If `memory.pf_particles` is empty: initialize via `pf_model.initialize()`
    - Else: load from `memory.pf_particles` and `memory.pf_weights`
  - Compute prior mean: `sum(particles) / len(particles)`
  - Predict step: `particles_pred = pf_model.predict(particles)`
  - Determine audience name: "interviewer" if context=True, "listener" if context=False
  - Extract measurement:
    - Prompt LLM: "Estimate {audience_name}'s internal evaluation from their response [0,1]"
    - Include goal description in prompt
    - Include audience's utterance (text + body)
    - Extract via regex: `r"([01](?:\.\d+)?)"`
    - Fallback to 0.5 if no match
    - Clamp to [0,1] range
  - Update step:
    - Use `obs_sigma = 0.03` (not the PF model's default 0.08)
    - Weight particles: `w = exp(-0.5 * ((measurement - particle) / obs_sigma)^2)`
    - Normalize weights
    - Compute ESS: `1.0 / sum(w^2)`
    - If ESS < 0.5 * num_particles: resample via systematic resampling
    - After resampling: reset weights to uniform (1/N)
  - Compute I_hat:
    - If weights available: `I_hat = sum(particle * weight)`
    - Else: `I_hat = mean(particles)`
  - Compute PE (signed):
    - If `pf_history` has >= 2 entries: `prev_I_hat = pf_history[-2]["I_hat"]`
    - Else if 1 entry: `prev_I_hat = prior_mean`
    - Else: `prev_I_hat = prior_mean`
    - `pe = prev_I_hat - I_hat` (signed, not absolute)
  - Store in memory:
    - `memory.pf_particles = particles_upd`
    - `memory.pf_weights = weights_upd`
    - `memory.pf_history.append({turn, prior_mean, I_hat, ess, resampled, measurement})`
    - `memory.add_pe_record(turn, audience_text, I_hat, pe)`
- **Key Design Decisions**:
  - Measurement extraction uses LLM (not direct access to I_t)
  - PE computed as signed change in belief (prev_I_hat - current_I_hat)
  - PF state persisted in memory component for next turn
  - Use obs_sigma=0.03 in update (more conservative than PF default)
  - Context determines audience name (interviewer vs listener)

#### 1.5 IMPEReflectionComponent
- **Purpose**: Generate reflection based on current belief state I_hat
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Dependencies**: IMPEMemoryComponent, LanguageModel
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Methods**:
  - `post_observe()`: After particle filter update, generate reflection
- **Logic**:
  - Get last I_hat from PF history
  - Prompt LLM: "What will you change next turn to improve goal achievement? Current belief: I_hat"
  - Include cultural norms and traits in prompt (if actor has them)
  - Store reflection in memory
- **Key Design Decisions**:
  - Reflection based on I_hat (belief), not true state
  - Include interview context if enabled

#### 1.6 IMPEActComponent
- **Purpose**: Generate utterance (dialogue + body language) based on belief, conversation history, and reflections
- **Type**: `ActingComponent`
- **Dependencies**: IMPEMemoryComponent, LanguageModel, CulturalNormsComponent (optional), PersonalityTraitsComponent (optional)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Methods**:
  - `get_action_attempt(context, action_spec)`: Generate utterance
    - Check if PF history is empty (first turn)
    - If empty: call `_act_first_turn()`
    - Else: call `_act_based_on_belief()`
- **Logic**:
  - **First Turn** (`_act_first_turn()`):
    - No belief history available
    - Get goal from memory
    - Determine audience name: "interviewer" if context=True, "listener" if context=False
    - Build prompt:
      - Include `_prompt_header()` (norms + traits if available)
      - Include goal name, description, role (if context), ideal
      - Prompt: "Produce a short utterance (one sentence) to accomplish the goal"
      - Format: "DIALOGUE: <text>\nBODY: <body language>"
    - Parse response (same as below)
    - Store utterance in memory
    - Return formatted string: `"DIALOGUE: {text}\nBODY: {body}"`

  - **Subsequent Turns** (`_act_based_on_belief()`):
    - Get recent conversation (last recent_k turns)
    - Get recent PF history (last recent_k entries)
    - Get recent reflections (last recent_k)
    - Get current I_hat: `pf_history[-1]["I_hat"]` or 0.5 if empty
    - Determine audience name: "interviewer" if context=True, "listener" if context=False
    - Build prompt:
      - Include `_prompt_header()`
      - Include goal name, description, role (if context), ideal
      - Include current I_hat: "Current belief about {audience_name}'s evaluation = {I_hat:.2f}"
      - Include recent conversation: `memory.format_conversation(recent_conv)`
      - Include recent I_hat history: formatted as "(turn {t}) I_hat={value:.2f}"
      - Include recent reflections
      - Prompt: "Produce a short utterance to improve goal achievement"
      - Format: "DIALOGUE: <text>\nBODY: <body language>"
    - Parse response:
      - Extract dialogue: `r"DIALOGUE:\s*(.*)"` - fallback to entire response
      - Extract body: `r"BODY:\s*(.*)"` - fallback to empty string
    - Store utterance in memory via `memory.add_utterance(turn, speaker, text, body)`
    - Return formatted string: `"DIALOGUE: {text}\nBODY: {body}"`
- **Key Design Decisions**:
  - First turn: simpler prompt without I_hat context
  - Subsequent turns: include full context (conversation, I_hat history, reflections)
  - Always include body language in output
  - Format action as structured string for game master parsing
  - Context determines audience name (interviewer vs listener)

#### 1.7 CulturalNormsComponent
- **Purpose**: Inject cultural norms into agent prompts and handle initialization
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **State**:
  - `norms`: List of `CulturalNorm` objects (name, description)
  - `initialized`: bool - track if initialization prompt has been sent
- **Methods**:
  - `get_norms_text()`: Format norms as prompt text
    - Format: `"CULTURAL NORMS YOU FOLLOW:\n" + "\n".join(f"- {n.name}: {n.description}") + "\n\n"`
    - Returns empty string if no norms
  - `initialize_norms(language_model, agent_name)`: Send one-time initialization prompt (NEW)
    - Only if norms exist and not already initialized
    - Prompt: "You are {agent_name}. You are in an alternative world in the year 3025 where there is a new set of cultural norms. In all your interactions, you must follow these cultural norms: {norms descriptions}. If you fail to do so, you will be unsuccessful in your interactions and perceived negatively by others. Always follow these norms strictly."
    - Call language model with this prompt
    - Set `initialized = True`
  - `get_state()` / `set_state()`: For checkpointing
- **Key Design Decisions**:
  - Separate component for modularity
  - Can be attached to any entity
  - Formats norms consistently for prompt injection
  - One-time initialization prompt sets up agent's understanding of norms
  - Initialization happens once, not per-turn

#### 1.8 PersonalityTraitsComponent
- **Purpose**: Inject personality traits with scores into agent prompts
- **Type**: `ContextComponent` (ActionSpecIgnored)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **State**:
  - `traits`: List of `PersonalityTrait` objects (name, assertion)
  - `trait_scores`: Dict[str, int] mapping trait names to scores (0-3)
- **Methods**:
  - `get_traits_text()`: Format traits with scores as prompt text
    - Format: `"YOUR PERSONALITY TRAITS, scored from 0 to 3. Each score has the following meaning: {0: False, not at all, 1: Slightly true, 2: Mainly true, 3: Very true.}:\n" + "\n".join(f"- {t.name} ({self.trait_scores.get(t.name, 'NA')} / 3): {t.assertion}") + "\n\n"`
    - Returns empty string if no traits
  - `get_state()` / `set_state()`: For checkpointing
- **Key Design Decisions**:
  - Separate component for modularity
  - Scores stored separately from trait definitions
  - Format: "Trait Name (score/3): assertion"
  - Score meanings documented in prompt text

### 2. Constants and Configuration

#### 2.1 Constants File
- **Location**: `projects/impression_management/constants.py`
- **Purpose**: Centralize all constants, defaults, and configuration data
- **Contents**:
  - **Cultural Norms** (17 items): All `CulturalNorm` objects with names and descriptions
    - "Stated purpose first", "Announced topics", "Direct, literal language", etc.
  - **Personality Traits** (11 items): All `PersonalityTrait` objects with names and assertions
    - "Detail-focused", "Avoids eye contact", "Not laid back", etc.
  - **Default Interview Role**: Multi-line string with Product Manager role definition
    - Includes responsibilities and evaluation criteria
  - **Default Agent Names**: `DEFAULT_ACTOR_NAME = "John"`, `DEFAULT_AUDIENCE_NAME = "Jane"`
  - **Default Parameters**:
    - `DEFAULT_NUM_PARTICLES = 200`
    - `DEFAULT_PROCESS_SIGMA = 0.03`
    - `DEFAULT_OBS_SIGMA = 0.08`
    - `DEFAULT_RECENT_K = 3`
    - `DEFAULT_TEMPERATURE = 0.2`
    - `DEFAULT_TOP_P = 0.9`
    - `DEFAULT_TURNS = 2`
    - `DEFAULT_SEED = 7`
  - **Trait Score Ranges**:
    - `AUDIENCE_TRAIT_SCORE_MIN = 2`, `AUDIENCE_TRAIT_SCORE_MAX = 3`
    - `ACTOR_TRAIT_SCORE_MIN = 0`, `ACTOR_TRAIT_SCORE_MAX = 1`
- **Usage**: Import constants in main script and other modules
- **Key Design Decisions**:
  - All constants in one file for easy modification
  - Can be extended with custom norms/traits
  - Defaults match original script

#### 2.2 Utility Functions Module
- **Location**: `projects/impression_management/utils.py`
- **Purpose**: Reusable utility functions for parsing, formatting, and data manipulation
- **Functions**:
  - `parse_index_list(s: str) -> List[int]`: Parse comma-separated 1-based indices to 0-based list
    - Handles empty strings, invalid entries
    - Returns empty list on error
  - `select_by_indices(full_list: List[Any], indices: List[int]) -> List[Any]`: Select items by indices
    - Validates indices are in range
  - `generate_trait_scores(rng: random.Random, trait_list: List[PersonalityTrait], is_audience: bool) -> Dict[str, int]`:
    - Audience: scores 2-3 (high traits)
    - Actor: scores 0-1 (low traits)
    - Uses RNG for reproducibility
  - `create_output_directory(save_dir: Optional[str]) -> str`: Create output directory
    - If `save_dir` is None: create `./temp/YYYY-MM-DD_HH-MM-SS/`
    - If `save_dir` provided: use it (create if doesn't exist)
    - Returns path string
  - `format_conversation(utterances: List[Utterance]) -> str`: Format utterances for prompts
    - Format: `"- [t={turn} {speaker}] {text}"` per utterance
    - Returns `"- (none)"` if empty
  - `parse_dialogue_and_body(response: str) -> Tuple[str, str]`: Parse LLM response
    - Extract dialogue: `r"DIALOGUE:\s*(.*)"` - fallback to entire response
    - Extract body: `r"BODY:\s*(.*)"` - fallback to empty string
    - Returns (dialogue, body) tuple
  - `extract_numeric_from_response(response: str, default: float = 0.5) -> float`: Extract number from LLM response
    - Regex: `r"([01](?:\.\d+)?)"` - matches 0.0 to 1.0
    - Fallback to default if no match
    - Returns clamped value in [0,1]
  - `clamp_to_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float`: Clamp value to range
- **Key Design Decisions**:
  - All parsing functions have fallbacks
  - All functions are pure (no side effects)
  - Error handling built into each function

#### 2.3 Plotting Module
- **Location**: `projects/impression_management/plotting.py`
- **Purpose**: Visualization functions for analysis
- **Functions**:
  - `plot_learning_dynamics(log: List[TurnLog], save_dir: str) -> None`: Generate all three plots
    - Calls individual plot functions
  - `plot_pe_trajectory(turns: List[int], pe_values: List[float], save_path: str) -> None`:
    - Plot PE vs turns (starts from turn 2, since PE needs previous I_hat)
    - X-axis: Turn, Y-axis: Prediction Error
    - Marker: "o", grid: True, title: "Prediction Error Across Turns"
    - Save as PNG, 200 DPI, tight bbox
  - `plot_belief_trajectories(turns: List[int], I_t: List[float], I_hat: List[float], save_path: str) -> None`:
    - Plot I_t and I_hat vs turns on same plot
    - I_t: marker "x", label "True I_t"
    - I_hat: marker "o", label "Estimated I_hat"
    - Legend, grid, title: "I_t and I_hat Across Turns"
    - Save as PNG, 200 DPI, tight bbox
  - `plot_learning_gain(turns: List[int], learning_gain: List[float], save_path: str) -> None`:
    - Plot learning gain vs turns
    - Learning gain = |delta I_hat| / (|PE| + epsilon), epsilon=1e-6
    - Marker: "s", grid: True, title: "Learning Gain Across Turns"
    - Save as PNG, 200 DPI, tight bbox
- **Dependencies**: matplotlib, numpy
- **Key Design Decisions**:
  - Modular functions for each plot type
  - Consistent styling (200 DPI, tight bbox)
  - Plots saved to specified directory

### 3. Custom Prefabs

#### 3.1 IMPEActorEntity Prefab
- **Purpose**: Entity prefab for actor (interviewee) role
- **Location**: `concordia/prefabs/entity/impression_management_actor.py`
- **Components** (initialization order matters):
  1. Instructions (goal description with interview context if enabled)
  2. IMPEMemoryComponent (must be first - others depend on it)
  3. CulturalNormsComponent (optional, typically empty for actor)
  4. PersonalityTraitsComponent (optional, typically has traits)
  5. IMPEActorParticleFilterComponent (depends on memory)
  6. IMPEReflectionComponent (depends on memory)
  7. IMPEActComponent (as act_component, depends on all above)
  8. ObservationToMemory (standard Concordia component)
- **Parameters**:
  - `name`: Entity name (default "John")
  - `goal_name`: Goal name (e.g., "competence")
  - `goal_description`: Goal description
  - `goal_role`: Interview role description (optional, None if no context)
  - `goal_ideal`: Ideal value (default 1.0)
  - `recent_k`: Window size (default 3)
  - `cultural_norms`: List of CulturalNorm objects (default empty - actor doesn't have norms)
  - `traits`: List of PersonalityTrait objects (default empty, but typically populated)
  - `trait_scores`: Dict[str, int] (default empty, but typically scores 0-1)
  - `context`: bool - enable interview context (default True)
  - `pf_num_particles`: int (default 200)
  - `pf_process_sigma`: float (default 0.03)
  - `pf_obs_sigma`: float (default 0.08, but actual update uses 0.03)
  - `seed`: int for RNG (default 0)
- **Initialization**:
  - Create ParticleFilter instance with parameters
  - Pass to IMPEActorParticleFilterComponent
  - Cultural norms initialization not needed (actor has no norms)

#### 3.2 IMPEAudienceEntity Prefab
- **Purpose**: Entity prefab for audience (interviewer) role
- **Location**: `concordia/prefabs/entity/impression_management_audience.py`
- **Components** (initialization order matters):
  1. Instructions (evaluation goal description)
  2. IMPEMemoryComponent (must be first - others depend on it)
  3. CulturalNormsComponent (typically populated for audience)
  4. PersonalityTraitsComponent (typically populated)
  5. IMPEAudienceEvaluationComponent (depends on memory, norms, traits)
  6. ObservationToMemory (standard)
- **Parameters**:
  - `name`: Entity name (default "Jane")
  - `goal_name`: Goal name (e.g., "evaluate_competence")
  - `goal_description`: Goal description
  - `goal_role`: Interview role description (optional, used if context=True)
  - `recent_k`: Window size (default 3)
  - `cultural_norms`: List of CulturalNorm objects (typically all 17 norms)
  - `traits`: List of PersonalityTrait objects (typically all 11 traits)
  - `trait_scores`: Dict[str, int] (typically scores 2-3 for audience)
  - `context`: bool - enable interview context (default True)
  - `seed`: int for RNG (default 0, not used for audience)
- **Initialization**:
  - After component creation, call `cultural_norms_component.initialize_norms(language_model, name)`
  - This sends one-time setup prompt to establish norms understanding
  - Only done if norms exist and not already initialized

#### 3.3 IMPEConversationGameMaster Prefab
- **Purpose**: Orchestrate impression management PE conversation flow
- **Location**: `concordia/prefabs/game_master/impression_management_pe.py`
- **Components**:
  - Instructions (conversation premise)
  - PlayerCharacters (actor and audience entities)
  - NextActing (fixed order: actor always speaks first, never alternates)
  - NextActionSpec
  - MakeObservation (passes actor's utterance to audience, audience's response to actor)
  - EventResolution (stores utterances as events, extracts body language)
  - Terminate (after max turns)
- **Special Logic**:
  - **Turn Structure**: Actor acts → Audience evaluates and responds → Actor updates particles → Actor reflects
  - **Each turn = 4 steps**: act, evaluate, update_particles, reflect
  - **Turn numbering**: 1-based (t=1, 2, 3, ...)
  - **Actor always initiates**: Never alternates - actor always speaks first in each turn
  - **Body language extraction**: Parse action string format "DIALOGUE: {text}\nBODY: {body}"
    - Use regex: `r"DIALOGUE:\s*(.*)"` and `r"BODY:\s*(.*)"`
    - Fallback: if DIALOGUE not found, use entire action as dialogue, body=""
  - **Observation formatting**:
    - To audience: `"Actor said: \"{text}\"\nBody language: \"{body}\""`
    - To actor: `"Audience said: \"{text}\"\nBody language: \"{body}\""`
  - **Turn tracking**: Include turn number in observation context for components
- **Key Design Decisions**:
  - Asymmetric turn structure (actor initiates, audience responds - never reversed)
  - Body language extracted and passed separately in observations
  - Turn number included in observation context
  - Fixed order ensures consistent flow

### 4. Simulation Structure

#### 4.1 Main Script Structure
- **Location**: `projects/impression_management/pe_conversation_concordia.py`
- **Structure**:
```python
# Imports
- Import constants from projects.impression_management.constants (or use relative import)
- Import utils from projects.impression_management.utils
- Import plotting from projects.impression_management.plotting
- Import components and prefabs from concordia

# CLI Argument Parsing
- Parse all arguments (see section 6.4 for complete list)
- Validate API key (exit if missing)
- Handle flags (--no_audience_norms, --no_traits, --no_context)

# Setup
- Load language model:
  - If --llm_type=openai: Use OpenAI API with retry logic
  - If --llm_type=local: Use Ollama local model
- Load embedder (sentence transformers)
- Load prefabs (including custom IMPE prefabs)

# Configuration
- Load cultural norms from constants (filter if --no_audience_norms)
- Load personality traits from constants (filter if --no_traits)
- Generate trait scores using utils.generate_trait_scores():
  - Audience: scores 2-3
  - Actor: scores 0-1
  - Use seed for reproducibility
- Load or create interview role (from constants if context enabled)
- Create goal objects (actor and audience)

# Entity Creation
- Create IMPEActorEntity instance:
  - Name from --actor_name (default "John")
  - Goal from actor_goal
  - Cultural norms: empty list (actor doesn't have norms)
  - Traits and trait_scores from generated scores
  - Context flag from --no_context
  - PF parameters from defaults or CLI
- Create IMPEAudienceEntity instance:
  - Name from --audience_name (default "Jane")
  - Goal from audience_goal
  - Cultural norms: all norms (unless --no_audience_norms)
  - Traits and trait_scores from generated scores
  - Context flag from --no_context
- Initialize cultural norms (one-time setup prompt) for audience if norms exist
- Create IMPEConversationGameMaster instance

# Simulation Setup
- Create Config with instances and prefabs
- Create Simulation with config, model, embedder
- Create output directory using utils.create_output_directory()

# Run
- Initialize simulation
- Run with max_steps = total_turns * 4 (each turn = 4 steps)
- Extract log data from entities (preferred) or raw_log (fallback)
- Pretty print turns using pretty_print_turns() function
- Generate plots using plotting.plot_learning_dynamics() (if matplotlib available)
- Save JSON to output directory

# Error Handling
- Try/except for LLM calls (retry with backoff)
- Try/except for file I/O
- Fallback to raw_log if component access fails
- Print errors but continue execution where possible
```

#### 4.2 Turn Flow in Concordia
**Turn Structure** (each turn = 4 steps, actor always initiates):
1. **Step 1 (Actor Act)**:
   - Game master selects actor (fixed: actor always speaks first)
   - Actor's IMPEActComponent generates utterance (dialogue + body)
   - First turn (t=1): Uses simpler prompt without I_hat context
   - Subsequent turns (t>=2): Uses full context (I_hat, conversation history, reflections)
   - Utterance stored in actor's memory
   - Utterance added to audience's memory (for context)

2. **Step 2 (Audience Evaluate)**:
   - Game master creates observation for audience: `"Actor said: \"{text}\"\nBody language: \"{body}\""`
   - Audience's IMPEAudienceEvaluationComponent processes observation
   - Generates true hidden state I_t ∈ [0,1] via LLM
   - Generates feedback response (dialogue + body language) matching I_t
   - Stores I_t and response in audience's evaluation_history and conversation

3. **Step 3 (Actor Update Particles)**:
   - Game master creates observation for actor: `"Audience said: \"{text}\"\nBody language: \"{body}\""`
   - Actor's IMPEActorParticleFilterComponent processes observation
   - Extracts measurement from audience's response via LLM
   - Updates particle filter (predict → weight → resample if needed)
   - Computes I_hat (posterior mean) and ESS
   - Computes PE = prev_I_hat - I_hat (signed)
   - Stores PF state and PE record in actor's memory

4. **Step 4 (Actor Reflect)**:
   - Actor's IMPEReflectionComponent generates reflection
   - Uses current I_hat from PF history
   - Stores reflection in actor's memory

5. **Repeat** for next turn (t+1)

**Key Points**:
- Turns are 1-based (start at t=1, not t=0)
- Actor always initiates, audience always responds (never alternates)
- Each turn = 4 simulation steps
- Total steps = total_turns * 4

#### 4.3 Data Flow
- **During Simulation**:
  - All data stored in IMPEMemoryComponent instances
  - PF state updated in real-time (particles, weights, history)
  - Components access memory via component key lookups
  - I_t stored in audience's evaluation_history during evaluation
  - Utterances stored in both actor's and audience's conversation lists
  - PE records stored in actor's pe_history (signed values)
  - Reflections stored in actor's reflections list
- **After Simulation**:
  - Extract data from entity components (preferred method):
    - Access IMPEMemoryComponent from each entity
    - Extract conversation, evaluation_history, pf_history, pe_history, reflections
    - Match by turn number to construct TurnLog entries
  - Fallback: parse raw_log if component extraction fails
  - Format as TurnLog dataclass (PE as absolute value)
  - Serialize to JSON (via asdict())
  - Save to output directory (timestamped if not specified)

### 5. Data Extraction and Logging

#### 5.1 Log Structure
- **Access Methods**:
  1. **Primary**: Extract directly from entity components
     ```python
     entities = {e.name: e for e in sim.get_entities()}
     actor = entities['Actor Name']
     audience = entities['Audience Name']

     # Get memory components
     actor_mem = actor.get_component('IMPE_Memory', type_=IMPEMemoryComponent)
     audience_mem = audience.get_component('IMPE_Memory', type_=IMPEMemoryComponent)

     # Extract data
     conversation = actor_mem.get_recent_conversation(k=total_turns)
     pf_history = actor_mem.get_pf_history(k=total_turns)
     pe_history = actor_mem.get_recent_pe_history(k=total_turns)
     reflections = actor_mem.get_recent_reflections(k=total_turns)
     ```
  2. **Fallback**: Parse raw_log from simulation
     - Access `raw_log` list passed to `sim.play()`
     - Parse step-by-step to extract component outputs
     - Match utterances to turns

#### 5.2 TurnLog Data Structure
```python
@dataclass
class TurnLog:
    time: str  # ISO timestamp (ISO format with 'Z' suffix)
    turn: int  # Turn number (1-based, starting at 1)
    speaker: str  # Actor name (always actor, never alternates)
    listener: str  # Audience name (always audience)
    speaker_text: str  # Actor's dialogue
    speaker_body: str  # Actor's body language
    audience_I: float  # True hidden state I_t (from audience evaluation)
    audience_text: str  # Audience's feedback dialogue
    audience_body: str  # Audience's body language
    actor_I_hat: float  # Actor's belief estimate (from PF history)
    actor_pe: float  # Prediction error (absolute value: |prev_I_hat - I_hat|)
    reflection_text: str  # Actor's reflection
    ess: float  # Effective sample size (from PF history)
```

**Note on PE**:
- In `PERecord`: PE is signed (prev_I_hat - current_I_hat) to show direction
- In `TurnLog`: PE is absolute value (|prev_I_hat - current_I_hat|) for plotting/analysis
- First turn: PE defaults to 1.0 if no previous I_hat available

#### 5.3 Extraction Logic
- **From Components** (Preferred):
  1. Get actor and audience entities by name
  2. Access IMPEMemoryComponent from each via component key lookup
  3. Extract conversation history from both (sorted by turn)
  4. Extract evaluation history from audience (contains I_t values)
  5. Extract PF history from actor (contains I_hat, ESS, etc.)
  6. Extract PE history from actor (contains signed PE)
  7. Extract reflections from actor
  8. For each turn (1-based, starting at 1):
     - Get actor's utterance from actor's conversation (where speaker=actor_name)
     - Get audience's I_t from audience's evaluation_history (match by turn)
     - Get audience's response from audience's conversation (where speaker=audience_name, turn matches)
     - Get actor's I_hat from actor's PF history (match by turn)
     - Get actor's PE: use absolute value from TurnLog computation (not signed from PERecord)
     - Get actor's ESS from actor's PF history (match by turn)
     - Get actor's reflection from actor's reflections (match by turn)
  9. Construct TurnLog entries with all fields
  10. Sort by turn number
- **From Raw Log** (Fallback):
  1. Iterate through raw_log steps
  2. Identify actor action steps (IMPEActComponent output)
  3. Identify audience evaluation steps (IMPEAudienceEvaluationComponent output - contains I_t)
  4. Identify particle filter update steps (IMPEActorParticleFilterComponent output - contains I_hat, ESS)
  5. Identify reflection steps (IMPEReflectionComponent output)
  6. Match by turn number and entity name
  7. Extract structured data
  8. Handle missing data with defaults (I_t=0.5, I_hat=0.5, PE=0.0, ESS=0.0)

#### 5.4 JSON Output
- **Format**: List of TurnLog dictionaries (via `asdict()`)
- **Location**: Specified by `--outfile` argument or timestamped directory
- **Encoding**: UTF-8, ensure_ascii=False for proper Unicode handling
- **Structure**:
```json
[
  {
    "time": "2024-01-01T12:00:00Z",
    "turn": 1,
    "speaker": "John",
    "listener": "Jane",
    "speaker_text": "...",
    "speaker_body": "...",
    "audience_I": 0.75,
    "audience_text": "...",
    "audience_body": "...",
    "actor_I_hat": 0.68,
    "actor_pe": 0.12,
    "reflection_text": "...",
    "ess": 145.3
  },
  ...
]
```

#### 5.5 Plotting and Visualization
- **Location**: `projects/impression_management/plotting.py` (separate module)
- **Function**: `plot_learning_dynamics(log: List[TurnLog], save_dir: str)`
- **Plots Generated**:
  1. **Prediction Error**: `pe.png` - PE across turns (starts from turn 2, since PE needs previous I_hat)
     - X-axis: Turn, Y-axis: Prediction Error
     - Marker: "o", grid: True
  2. **Belief Trajectories**: `delta_I.png` - I_t (true) vs I_hat (estimated) across turns
     - I_t: marker "x", label "True I_t"
     - I_hat: marker "o", label "Estimated I_hat"
     - Legend, grid: True
  3. **Learning Gain**: `learning_gain.png` - |delta I_hat| / (|PE| + epsilon) across turns
     - Learning gain = |delta I_hat| / (|PE| + 1e-6)
     - Marker: "s", grid: True
- **Dependencies**: matplotlib, numpy
- **Output**: PNG files in save directory (200 DPI, tight bbox)
- **Usage**: Import from plotting module, call after data extraction

### 6. Implementation Files

#### 6.1 Core Components
- **File**: `concordia/components/agent/impression_management_pe.py`
- **Contents**:
  - Data classes: `Goal`, `Utterance`, `PERecord`, `ReflectionRecord`, `EvaluationRecord` (NEW), `CulturalNorm`, `PersonalityTrait`
  - `ParticleFilter` utility class
  - `IMPEMemoryComponent`
  - `IMPEAudienceEvaluationComponent`
  - `IMPEActorParticleFilterComponent`
  - `IMPEReflectionComponent`
  - `IMPEActComponent`
  - `CulturalNormsComponent`
  - `PersonalityTraitsComponent`
  - Component key constants (e.g., `DEFAULT_IMPE_MEMORY_COMPONENT_KEY = 'IMPE_Memory'`)

#### 6.2 Entity Prefabs
- **File**: `concordia/prefabs/entity/impression_management_actor.py`
- **Contents**: `IMPEActorEntity__Entity` prefab class
- **File**: `concordia/prefabs/entity/impression_management_audience.py`
- **Contents**: `IMPEAudienceEntity__Entity` prefab class

#### 6.3 Game Master Prefab
- **File**: `concordia/prefabs/game_master/impression_management_pe.py`
- **Contents**: `IMPEConversationGameMaster__GameMaster` prefab class

#### 6.4 Main Script
- **File**: `projects/impression_management/pe_conversation_concordia.py`
- **Contents**:
  - Imports from constants, utils, plotting, components, prefabs
  - `TurnLog` dataclass
  - `extract_turn_data_from_entities()` function
  - `extract_turn_data_from_log()` function (fallback)
  - `pretty_print_turns()` function (NEW)
  - `main()` function with complete CLI argument parsing
  - Language model setup (OpenAI with retry logic, or local Ollama)
  - Entity/prefab creation
  - Simulation orchestration
  - Data extraction and saving
- **CLI Arguments** (complete list):
  - `--turns` (int, default=2): Total turns in dialogue
  - `--model` (str, default="gpt-4o-mini"): OpenAI model name
  - `--temperature` (float, default=0.2): Sampling temperature
  - `--top_p` (float, default=0.9): Top-p nucleus sampling
  - `--window` (int, default=3): Recent K turns to condition on
  - `--outfile` (str, default="pe_conversation_log.json"): JSON output filename
  - `--no_audience_norms` (flag): Disable cultural norms for audience
  - `--no_traits` (flag): Disable personality traits
  - `--no_context` (flag): Disable interview context
  - `--seed` (int, default=7): Random seed for reproducibility
  - `--save_dir` (str, default=None): Output directory (creates timestamped if None)
  - `--actor_name` (str, default="John"): Actor/interviewee name
  - `--audience_name` (str, default="Jane"): Audience/interviewer name
  - `--llm_type` (str, default="openai"): LLM type: "openai" or "local"
  - `--local_model` (str, default="llama3.1:8b"): Local model name (for Ollama)
- **Error Handling**:
  - API key validation (exit if missing)
  - LLM retry logic: exponential backoff (2^(attempt-1) seconds, max 8s), max 3 retries
  - Timeout: 30s for OpenAI, 120s for local
  - Parsing fallbacks: default values on failure
  - File I/O: try/except with error messages
  - Component access: fallback to raw_log parsing

#### 6.5 Constants File
- **File**: `projects/impression_management/constants.py`
- **Contents**:
  - All 17 cultural norms (as `CulturalNorm` objects)
  - All 11 personality traits (as `PersonalityTrait` objects)
  - Default interview role (Product Manager, multi-line string)
  - Default agent names (`DEFAULT_ACTOR_NAME`, `DEFAULT_AUDIENCE_NAME`)
  - All default parameters (num_particles, sigmas, recent_k, temperature, top_p, etc.)
  - Trait score ranges (audience: 2-3, actor: 0-1)

#### 6.6 Utils Module
- **File**: `projects/impression_management/utils.py`
- **Contents**:
  - `parse_index_list()`: Parse comma-separated indices
  - `select_by_indices()`: Select items by indices
  - `generate_trait_scores()`: Generate trait scores with RNG
  - `create_output_directory()`: Create timestamped directory if needed
  - `format_conversation()`: Format utterances for prompts
  - `parse_dialogue_and_body()`: Parse LLM response
  - `extract_numeric_from_response()`: Extract number from LLM response
  - `clamp_to_range()`: Clamp value to [0,1]

#### 6.7 Plotting Module
- **File**: `projects/impression_management/plotting.py`
- **Contents**:
  - `plot_learning_dynamics()`: Main function (calls individual plots)
  - `plot_pe_trajectory()`: Plot PE vs turns
  - `plot_belief_trajectories()`: Plot I_t and I_hat vs turns
  - `plot_learning_gain()`: Plot learning gain vs turns

#### 6.8 Tests
- **File**: `concordia/components/agent/impression_management_pe_test.py`
- **Contents**: Unit tests for all components
- **File**: `concordia/prefabs/entity/impression_management_actor_test.py`
- **Contents**: Integration tests for actor prefab
- **File**: `concordia/prefabs/entity/impression_management_audience_test.py`
- **Contents**: Integration tests for audience prefab

#### 6.9 Documentation
- **File**: `docs/impression_management_pe_components.md`
- **Contents**: Detailed documentation of components, usage examples, API reference (framework documentation)
- **File**: `projects/impression_management/docs/particle_filter.md`
- **Contents**: Particle filter documentation (project-specific)

### 7. Key Challenges & Solutions

#### Challenge 1: Particle Filter State Management
- **Problem**: PF state (particles, weights, history) must persist across turns
- **Solution**: Store in IMPEMemoryComponent with proper serialization in `get_state()`/`set_state()`
- **Implementation**: Use lists for particles/weights, dicts for history entries

#### Challenge 2: Asymmetric Turn Structure
- **Problem**: Actor and audience have different behaviors (actor acts, audience evaluates)
- **Solution**: Separate prefabs (IMPEActorEntity vs IMPEAudienceEntity) with different component configurations
- **Implementation**: Game master uses fixed order (actor → audience) with appropriate observation routing

#### Challenge 3: Body Language Extraction
- **Problem**: Actions include both dialogue and body language, must be parsed and passed separately
- **Solution**: IMPEActComponent formats output as "DIALOGUE: <text>\nBODY: <body>", game master parses and includes in observation
- **Implementation**: Regex parsing in game master's EventResolution component

#### Challenge 4: True Hidden State Access
- **Problem**: Audience generates I_t but actor must infer it from response
- **Solution**: I_t stored in audience's memory during evaluation, actor uses LLM to extract measurement from response
- **Implementation**: IMPEAudienceEvaluationComponent stores I_t in memory, IMPEActorParticleFilterComponent infers from observation

#### Challenge 5: Cultural Norms and Traits Integration
- **Problem**: Norms and traits must be injected into all LLM prompts consistently
- **Solution**: Separate components (CulturalNormsComponent, PersonalityTraitsComponent) that format prompt text
- **Implementation**: Components provide `get_norms_text()` and `get_traits_text()` methods, other components call these when building prompts

#### Challenge 6: ESS and PF Diagnostics
- **Problem**: Need to track PF diagnostics (ESS, resampling flags) for analysis
- **Solution**: Store full PF history in IMPEMemoryComponent including all diagnostic information
- **Implementation**: PF history entries include: turn, prior_mean, I_hat, ess, resampled, measurement

#### Challenge 7: Interview Context Switching
- **Problem**: System should work with or without interview context
- **Solution**: `context` boolean parameter in prefabs, conditional prompt text generation
- **Implementation**: Check `context` flag in components, adjust prompts and role names accordingly

### 8. Testing Strategy

#### 8.1 Unit Tests
1. **ParticleFilter**:
   - Test initialization
   - Test predict step (particles stay in [0,1])
   - Test update step (weights normalized correctly)
   - Test resampling (ESS threshold triggers)
   - Test systematic resampling algorithm

2. **IMPEMemoryComponent**:
   - Test add/retrieve utterances (with body language)
   - Test add/retrieve PE records
   - Test add/retrieve reflections
   - Test PF state storage/retrieval
   - Test state serialization (get_state/set_state)

3. **IMPEAudienceEvaluationComponent**:
   - Test I_t extraction from LLM response
   - Test response generation matching I_t
   - Test body language parsing
   - Test cultural norms injection

4. **IMPEActorParticleFilterComponent**:
   - Test measurement extraction from observation
   - Test PF update flow (predict → weight → resample)
   - Test I_hat computation
   - Test PE calculation

5. **IMPEReflectionComponent**:
   - Test reflection generation
   - Test I_hat context inclusion

6. **IMPEActComponent**:
   - Test first turn act() (no belief history)
   - Test subsequent act_based_on_belief() (with I_hat)
   - Test body language generation
   - Test context formatting (conversation, PF history, reflections)

#### 8.2 Integration Tests
1. **Full Turn Cycle**:
   - Create actor and audience entities
   - Simulate one turn: act → evaluate → update_particles → reflect
   - Verify all data stored correctly
   - Verify PF state updated

2. **Multi-Turn Conversation**:
   - Run 3-5 turns
   - Verify conversation history accumulates
   - Verify PF history tracks correctly
   - Verify PE computed correctly

3. **Cultural Norms and Traits**:
   - Create entities with norms and traits
   - Verify prompts include formatted norms/traits
   - Verify behavior changes with different trait scores

#### 8.3 End-to-End Tests
1. **Full Simulation**:
   - Run complete simulation (6 turns)
   - Extract data from entities
   - Verify TurnLog structure
   - Verify JSON output format
   - Compare with original system output

2. **Plot Generation**:
   - Run simulation
   - Generate plots
   - Verify files created
   - Verify plot data matches log data

### 9. Implementation Order

1. **Phase 1: Foundation**
   - Create data classes (Goal, Utterance, PERecord, etc.)
   - Implement ParticleFilter utility class
   - Implement IMPEMemoryComponent
   - Write unit tests for foundation

2. **Phase 2: Core Components**
   - Implement IMPEAudienceEvaluationComponent
   - Implement IMPEActorParticleFilterComponent
   - Implement IMPEReflectionComponent
   - Implement IMPEActComponent
   - Write unit tests for each component

3. **Phase 3: Supporting Components**
   - Implement CulturalNormsComponent
   - Implement PersonalityTraitsComponent
   - Write unit tests

4. **Phase 4: Prefabs**
   - Implement IMPEActorEntity prefab
   - Implement IMPEAudienceEntity prefab
   - Implement IMPEConversationGameMaster prefab
   - Write integration tests

5. **Phase 5: Main Script**
   - Implement main script with CLI
   - Implement data extraction functions
   - Implement plotting function
   - Add cultural norms and traits constants

6. **Phase 6: Testing & Refinement**
   - Run end-to-end tests
   - Compare with original system
   - Fix bugs and edge cases
   - Optimize performance
   - Write documentation

### 10. Data Access Patterns

#### 10.1 During Simulation
- Components access memory via entity's `get_component()` method
- Memory component key: `'IMPE_Memory'` (constant)
- Other components access memory by key lookup
- PF state updated in-place in memory component

#### 10.2 After Simulation
- **Primary Method**: Direct component access
  ```python
  # Get entities
  entities = {e.name: e for e in sim.get_entities()}
  actor = entities['Actor Name']

  # Get memory component
  mem = actor.get_component('IMPE_Memory', type_=IMPEMemoryComponent)

  # Extract data
  conversation = mem.get_recent_conversation(k=total_turns)
  pf_history = mem.get_pf_history(k=total_turns)
  ```
- **Fallback Method**: Raw log parsing
  ```python
  # Parse raw_log
  for step in raw_log:
      # Extract component outputs by key
      if 'IMPE_Act' in step:
          # Parse actor action
      if 'IMPE_Audience_Evaluation' in step:
          # Parse audience evaluation
  ```

#### 10.3 Logging Architecture
- **In-Memory Storage**: All data in IMPEMemoryComponent instances
- **Simulation Log**: Concordia's raw_log captures component outputs at each step
- **Extracted Log**: TurnLog list created post-simulation from components or raw_log
- **Persistent Storage**: JSON file with full TurnLog data
- **Visualization**: Plots generated from TurnLog data

### 11. File Organization Summary

```
concordia/
├── components/
│   └── agent/
│       ├── impression_management_pe.py          # Components + ParticleFilter + Data classes
│       └── impression_management_pe_test.py      # Unit tests
├── prefabs/
│   ├── entity/
│   │   ├── impression_management_actor.py        # Actor prefab
│   │   ├── impression_management_audience.py     # Audience prefab
│   │   ├── impression_management_actor_test.py   # Actor tests
│   │   └── impression_management_audience_test.py # Audience tests
│   └── game_master/
│       └── impression_management_pe.py          # Game master prefab
projects/
└── impression_management/
    ├── pe_conversation_concordia.py              # Main script
    ├── constants.py                               # All constants (norms, traits, defaults)
    ├── utils.py                                   # Utility functions
    ├── plotting.py                                # Plotting functions
    └── docs/                                      # Project documentation
        └── particle_filter.md                    # Particle filter documentation
docs/
└── impression_management_pe_components.md       # Framework component documentation
```

### 12. Notes on Concordia Integration

- **Component Keys**: Use descriptive constants (e.g., `DEFAULT_IMPE_MEMORY_COMPONENT_KEY = 'IMPE_Memory'`)
- **State Management**: All components implement `get_state()`/`set_state()` for checkpointing
- **Logging**: Use `ComponentWithLogging` base class for automatic logging integration
- **Type Hints**: Full type annotations for all methods and classes
- **Error Handling**: Graceful fallbacks (e.g., default I_t=0.5 if parsing fails)
- **Documentation**: Docstrings for all public methods following Google style

## 13. Additional Implementation Details

### 13.1 Regex Patterns
All regex patterns used in the implementation:
- **Numeric extraction**: `r"([01](?:\.\d+)?)"` - matches 0.0 to 1.0
- **Dialogue extraction**: `r"DIALOGUE:\s*(.*)"` - extracts dialogue text
- **Body extraction**: `r"BODY:\s*(.*)"` - extracts body language

### 13.2 Prompt Construction Pattern
Standard pattern for building prompts:
```
{prompt_header()}  # Norms + traits if available
{context_prompt}    # Interview context if enabled
{goal_description} # Goal name, description, role, ideal
{instruction}      # Specific task instruction
{data}             # Conversation history, I_hat, etc.
{format_instruction} # Output format specification
```

### 13.3 Dependencies
Required Python packages:
- `concordia` (framework)
- `openai` (for OpenAI API)
- `requests` (for local Ollama)
- `sentence-transformers` (for embedder)
- `numpy` (for numerical operations)
- `matplotlib` (optional, for plotting)
- `dataclasses` (standard library)
- `typing` (standard library)
- `random` (standard library)
- `re` (standard library)
- `json` (standard library)
- `datetime` (standard library)
- `os` (standard library)
- `sys` (standard library)
- `math` (standard library)

### 13.4 Turn Numbering Convention
- **1-based**: Turns start at 1, not 0
- **First turn**: t=1, uses `act()` method (no I_hat history)
- **Subsequent turns**: t>=2, use `act_based_on_belief()` method
- **All data structures**: Use 1-based turn numbers consistently

### 13.5 PE Computation Details
- **In PERecord**: PE is signed (prev_I_hat - current_I_hat)
  - Positive: belief decreased
  - Negative: belief increased
- **In TurnLog**: PE is absolute value (|prev_I_hat - current_I_hat|)
  - Used for plotting and analysis
  - First turn: defaults to 1.0 if no previous I_hat

### 13.6 Context-Based Name Switching
- **If context=True**:
  - Actor name in prompts: "interviewee"
  - Audience name in prompts: "interviewer"
- **If context=False**:
  - Actor name in prompts: "partner"
  - Audience name in prompts: "listener"
- Applied consistently across all components

### 13.7 Memory Serialization
All memory state must be JSON-serializable:
- Lists of floats (particles, weights)
- Lists of dataclasses (conversation, pe_history, reflections, evaluation_history)
- Dicts (pf_history entries, goal)
- Use `asdict()` for dataclasses
- Standard JSON serialization for other types

## 14. Reuse and Extension Strategy

### Existing Components Analysis
- **Existing PE System**: `concordia/components/agent/pe_conversation.py` provides basic PE conversation components
- **Key Difference**: Existing system uses simple LLM estimation; impression management uses particle filter
- **Strategy**: Extend existing data classes, create new components for different algorithms

### Data Class Extensions
- **`Goal`**: Add `role: str | None = None` to existing class in `pe_conversation.py`
- **`Utterance`**: Add `body: str = ""` to existing class in `pe_conversation.py`
- **Backward Compatibility**: New fields are optional, existing code continues to work

### Component Inheritance
- **`IMPEMemoryComponent`**: Inherits from `PEMemoryComponent`, adds:
  - Particle filter state (particles, weights, history)
  - Evaluation history (for I_t storage)
  - `format_conversation()` method
  - Overrides `get_state()` / `set_state()` for new fields

### New Components (Cannot Reuse)
- **Estimation**: Different algorithm (particle filter vs simple LLM)
- **Reflection**: Based on I_hat (belief) not PE value
- **Act**: Handles first turn differently, includes body language, uses particle filter belief
- **Cultural Norms/Traits**: No existing equivalent

### Import Dependencies
```python
# Required modification to pe_conversation.py:
# - Add role field to Goal
# - Add body field to Utterance

# In impression_management_pe.py:
from concordia.components.agent.pe_conversation import (
    Goal, Utterance, PERecord, ReflectionRecord, PEMemoryComponent
)
```

## 15. Implementation Completeness Checklist

This plan ensures full replication of `projects/impression_management/pe_conversation_openai.py`:

### Components ✓
- [x] IMPEMemoryComponent with evaluation_history
- [x] ParticleFilter utility class
- [x] IMPEAudienceEvaluationComponent (generates I_t)
- [x] IMPEActorParticleFilterComponent (updates PF, computes I_hat)
- [x] IMPEReflectionComponent
- [x] IMPEActComponent (handles first turn vs subsequent turns)
- [x] CulturalNormsComponent (with initialization)
- [x] PersonalityTraitsComponent

### Prefabs ✓
- [x] IMPEActorEntity (with all components)
- [x] IMPEAudienceEntity (with norms initialization)
- [x] IMPEConversationGameMaster (asymmetric turn structure)

### Data Structures ✓
- [x] Goal (with role field)
- [x] Utterance (with body language)
- [x] PERecord (signed PE)
- [x] ReflectionRecord
- [x] EvaluationRecord (NEW - for I_t storage)
- [x] CulturalNorm
- [x] PersonalityTrait
- [x] TurnLog (with all fields including ESS)

### Modular Structure ✓
- [x] Constants file (all norms, traits, defaults)
- [x] Utils module (all utility functions)
- [x] Plotting module (separate from main script)
- [x] Main script (orchestration only)

### Features ✓
- [x] Particle filter with ESS tracking
- [x] Cultural norms (17 items)
- [x] Personality traits (11 items, 0-3 scoring)
- [x] Interview context (optional)
- [x] Body language in utterances
- [x] True hidden state I_t generation
- [x] LLM measurement extraction
- [x] PE computation (signed and absolute)
- [x] Reflection generation
- [x] Plotting (3 plots)
- [x] Timestamped directory creation
- [x] Pretty print output
- [x] Local LLM support (Ollama)
- [x] LLM retry logic

### CLI Arguments ✓
- [x] All 13 arguments from original script
- [x] Default values specified
- [x] Flag handling documented

### Error Handling ✓
- [x] API key validation
- [x] LLM retry with backoff
- [x] Parsing fallbacks
- [x] File I/O error handling
- [x] Component access fallbacks

### Implementation Details ✓
- [x] Regex patterns specified
- [x] Prompt construction patterns
- [x] Observation formats
- [x] Turn numbering (1-based)
- [x] Component initialization order
- [x] Memory serialization requirements
- [x] Context-based name switching
- [x] First turn handling

This plan provides a comprehensive roadmap for implementing the impression management PE conversation system in the Concordia framework, with detailed specifications for components, data flow, file organization, and all implementation details needed to fully replicate the original script with a modular structure.
