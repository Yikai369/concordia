# Review of Impression Management PE Conversation Plan

## Overview

This review ensures the implementation plan can fully replicate `projects/impression_management/pe_conversation_openai.py` with a modular structure. The plan should support all features, maintain separation of concerns, and place constants in a dedicated file.

## Missing or Incomplete Elements

### 1. I_t Storage in Audience Memory
**Issue**: The plan states I_t is stored in audience memory, but the original code returns I_t from `audience_evaluate_and_respond()` without explicitly storing it in memory. It's only stored in TurnLog.

**Recommendation**:
- Clarify that I_t should be stored in IMPEMemoryComponent as part of the evaluation record
- Add method: `add_evaluation_record(turn, I_t, utterance)` to IMPEMemoryComponent
- Or store I_t in a separate `evaluation_history` list in memory

### 2. ConversationStudy Wrapper Class
**Issue**: The original code uses a `ConversationStudy` class that orchestrates the conversation loop. The plan doesn't clearly map this to Concordia's simulation structure.

**Recommendation**:
- Clarify that `ConversationStudy.run()` logic maps to the simulation's turn flow
- The timestamped directory creation logic should be in the main script, not a separate class
- Document how the study's `save_dir` logic translates to Concordia

### 3. Timestamped Directory Creation
**Issue**: The plan mentions `save_dir` but doesn't detail the automatic timestamped directory creation.

**Recommendation**:
- Add to main script: if `save_dir` is None, create `./temp/YYYY-MM-DD_HH-MM-SS/`
- Store plots and JSON in this directory
- Document this behavior in the main script section

### 4. Utility Functions
**Issue**: The plan mentions utility functions but doesn't specify where they go or their full implementation.

**Missing functions**:
- `parse_index_list(s: str) -> List[int]`: Parse comma-separated indices (for selecting norms/traits)
- `select_by_indices(full_list, indices)`: Select items by indices
- `generate_trait_scores(rng, trait_list, is_audience)`: Generate trait scores (audience: 2-3, actor: 0-1)

**Recommendation**:
- Add section: "Utility Functions" in main script
- Place in `examples/impression_management_pe_conversation.py` or separate utils module
- Document their purpose and usage

### 5. Local LLM Support
**Issue**: The original code supports both OpenAI API and local Ollama models via `make_local_llm()`. The plan only mentions OpenAI.

**Recommendation**:
- Add `make_local_llm()` function to main script or utils
- Document how to switch between OpenAI and local models
- Add CLI argument: `--llm_type` (openai/local)
- Add CLI argument: `--local_model` (for Ollama model name)

### 6. LLM Retry Logic
**Issue**: The OpenAI LLM wrapper has retry logic with exponential backoff (max_retries=3). The plan doesn't mention this.

**Recommendation**:
- Document retry logic in language model setup
- Specify: exponential backoff (2^(attempt-1) seconds, max 8s)
- Handle timeout (default 30s)
- This should be in Concordia's language model wrapper, but document if custom wrapper needed

### 7. Cultural Norms Initialization
**Issue**: The `initialize_cultural_norms()` method sends a special one-time prompt to "set" the norms. The plan mentions norms but doesn't detail this initialization step.

**Recommendation**:
- Add to IMPEAudienceEntity prefab: call initialization on entity creation
- Or add to CulturalNormsComponent: `initialize_norms()` method that sends setup prompt
- Document that this is a one-time setup, not per-turn

### 8. Prompt Header Method
**Issue**: The `_prompt_header()` method formats norms and traits consistently. The plan mentions this but implementation details could be clearer.

**Recommendation**:
- Clarify that CulturalNormsComponent and PersonalityTraitsComponent should provide formatted text
- Document the exact format: "CULTURAL NORMS YOU FOLLOW:\n- Name: Description\n\n"
- Document trait format: "Trait Name (score/3): assertion"
- Show how components combine their outputs

### 9. Interview Role Configuration
**Issue**: The Product Manager role is hardcoded in the original script. The plan should specify how roles are configured.

**Recommendation**:
- Add to configuration: `interview_role` parameter (string or None)
- Default: Product Manager role (as in original)
- Allow custom role definitions via CLI or config file
- Document role format (multi-line string with responsibilities and evaluation criteria)

### 10. PE Computation Discrepancy
**Issue**: In `ConversationStudy.run()`, PE is computed as `abs(prev_I_hat - I_hat)`, but in `actor_update_particles()` it's `prev_I_hat - I_hat` (signed). The plan should clarify which is used.

**Recommendation**:
- Clarify: PE in TurnLog should be absolute value (for plotting/analysis)
- PE in PERecord can be signed (for direction of change)
- Document both computations and their purposes

### 11. First Turn Special Handling
**Issue**: The plan mentions first turn uses `act()` but doesn't detail the `I_hat_prev = None` logic and how it affects behavior.

**Recommendation**:
- Document: First turn has no belief history, so `I_hat_prev = None`
- IMPEActComponent should check if PF history is empty
- If empty, use simpler prompt without I_hat context
- Clarify this in IMPEActComponent logic section

### 12. Actor Always Speaks First
**Issue**: The plan mentions asymmetric roles but doesn't explicitly state that actor ALWAYS initiates (never alternates).

**Recommendation**:
- Clarify in turn flow: Actor always speaks first in each turn
- Audience always responds (never initiates)
- This is different from alternating conversation patterns
- Document in IMPEConversationGameMaster section

### 13. Body Language Parsing Details
**Issue**: The plan mentions body language extraction but doesn't detail the regex parsing logic.

**Recommendation**:
- Document exact regex patterns: `r"DIALOGUE:\s*(.*)"` and `r"BODY:\s*(.*)"`
- Fallback behavior: if DIALOGUE not found, use entire response as dialogue
- If BODY not found, use empty string
- Show parsing logic in IMPEActComponent and IMPEAudienceEvaluationComponent

### 14. Measurement Extraction Details
**Issue**: The plan mentions LLM extracts measurement but doesn't detail the regex and fallback.

**Recommendation**:
- Document regex: `r"([01](?:\.\d+)?)"` - matches 0.0 to 1.0
- Fallback: if no match, use 0.5
- Clamp to [0,1] range
- Show this in IMPEActorParticleFilterComponent

### 15. Cultural Norms List
**Issue**: The plan doesn't include the full list of 17 cultural norms from the original code.

**Recommendation**:
- Add section: "Cultural Norms Constants" in main script
- List all 17 norms with names and descriptions
- Document that these are defaults, can be customized
- Consider making configurable via file or CLI

### 16. Personality Traits List
**Issue**: The plan doesn't include the full list of 11 personality traits.

**Recommendation**:
- Add section: "Personality Traits Constants" in main script
- List all 11 traits with names and assertions
- Document scoring system: 0-3 scale with meanings
- Document default scoring: audience 2-3, actor 0-1

### 17. CLI Arguments
**Issue**: The plan mentions CLI but doesn't list all arguments from the original.

**Missing arguments**:
- `--no_audience_norms`: Disable cultural norms for audience
- `--no_traits`: Disable personality traits
- `--no_context`: Disable interview context
- `--seed`: Random seed for reproducible trait scoring
- `--save_dir`: Directory for outputs (with timestamped default)

**Recommendation**:
- Add complete CLI arguments section
- Document all flags and their effects
- Show default values

### 18. Error Handling
**Issue**: The plan mentions graceful fallbacks but doesn't detail all error cases.

**Recommendation**:
- Document: API key validation
- LLM response parsing failures (fallback values)
- File I/O errors
- Component access failures (fallback to raw_log parsing)
- Add error handling section

### 19. Dependencies
**Issue**: The plan doesn't list all required dependencies.

**Missing**:
- `matplotlib` (for plotting)
- `numpy` (for numerical operations)
- `openai` (for OpenAI API)
- `requests` (for local LLM)
- `sentence-transformers` (for embedder, though this is Concordia standard)

**Recommendation**:
- Add dependencies section
- Create `requirements.txt` or update existing one
- Document optional dependencies (matplotlib for plotting)

### 20. Observation Format Details
**Issue**: The plan mentions observation format but doesn't show exact string format.

**Recommendation**:
- Show exact format: `"Actor said: \"{text}\"\nBody language: \"{body}\""`
- Show how game master constructs these
- Document parsing logic in components

### 21. PF History Access Pattern
**Issue**: The plan mentions PF history but doesn't show how to access previous I_hat for PE computation.

**Recommendation**:
- Document: Access `pf_history[-1]["I_hat"]` for current, `pf_history[-2]["I_hat"]` for previous
- Handle edge case: if only one entry, use `prior_mean` as previous
- Show this logic in IMPEActorParticleFilterComponent

### 22. Pretty Print Output
**Issue**: The plan doesn't mention the pretty-printed console output format.

**Recommendation**:
- Add to main script: pretty print function
- Format: `[t={turn}] {speaker} → {listener}: {text} [body: {body}]`
- Show I_t, I_hat, PE, reflection in formatted output
- Document this as optional but useful for debugging

### 23. Random Seed Usage
**Issue**: The plan mentions seed but doesn't detail how it's used for reproducibility.

**Recommendation**:
- Document: Seed used for trait score generation
- Seed passed to ParticleFilter RNG
- Seed passed to Agent RNG
- Ensure reproducibility across runs with same seed

### 24. Temperature and Top-p Parameters
**Issue**: The plan doesn't mention LLM sampling parameters.

**Recommendation**:
- Document default: temperature=0.2, top_p=0.9 (conservative for reduced variance)
- Add to CLI arguments
- Explain why conservative values are used (reduce variance in evaluations)

### 25. Window Size (recent_k)
**Issue**: The plan mentions recent_k but doesn't detail how it's used throughout.

**Recommendation**:
- Document: Used for conversation history window
- Used for PF history window in act_based_on_belief
- Used for PE history window
- Default: 3 turns
- Show usage in all relevant components

### 26. Conversation Formatting Method
**Issue**: The `format_conversation()` method formats utterances for prompts. Not mentioned in plan.

**Recommendation**:
- Add to IMPEMemoryComponent: `format_conversation(utterances: List[Utterance]) -> str`
- Format: `"- [t={turn} {speaker}] {text}"` per utterance
- Used in prompts to show recent conversation history
- Document in IMPEMemoryComponent methods

### 27. Recent Conversation Helper
**Issue**: The `recent_conversation(k)` method retrieves last k utterances. Plan mentions but doesn't show implementation.

**Recommendation**:
- Document: Returns `self.memory.conversation[-k:]` if k is None, uses `self.recent_k`
- Used throughout for context windows
- Show in IMPEMemoryComponent methods

### 28. Actor Name Context Switching
**Issue**: Actor name changes based on context ("partner" vs "interviewee"). Plan doesn't detail this.

**Recommendation**:
- Document: If `context=True`, use "interviewee"/"interviewer"
- If `context=False`, use "partner"/"listener"
- This affects all prompts in IMPEAudienceEvaluationComponent and IMPEActorParticleFilterComponent
- Show conditional logic in component implementations

### 29. Goal Role Field
**Issue**: Goal has a `role` field for interview context. Plan mentions but doesn't show how it's used in prompts.

**Recommendation**:
- Document: `goal.role` is multi-line string with role description
- Included in prompts when `context=True`
- Format: "You are interviewing for a candidate for the following role: {role}"
- Show in all components that use goal

### 30. ConversationStudy Save Directory Logic
**Issue**: The `ConversationStudy` class creates timestamped directories. Plan doesn't show this logic.

**Recommendation**:
- Add to main script: `create_output_directory(save_dir: Optional[str]) -> str`
- If `save_dir` is None: create `./temp/YYYY-MM-DD_HH-MM-SS/`
- Return path for use in plotting and JSON saving
- Document this utility function

### 31. Plot Learning Dynamics Details
**Issue**: The plan mentions plotting but doesn't detail the exact plot generation logic.

**Recommendation**:
- Document: Three plots generated
- Plot 1: PE vs turns (starts from turn 2, since PE needs previous I_hat)
- Plot 2: I_t and I_hat vs turns (both on same plot with different markers)
- Plot 3: Learning gain = |delta I_hat| / |PE| (with epsilon=1e-6 to avoid division by zero)
- All plots saved as PNG with 200 DPI, tight bbox
- Show exact matplotlib code structure

### 32. Turn Numbering
**Issue**: Original uses 1-based turn numbering (t=1, 2, 3...). Plan should specify this.

**Recommendation**:
- Document: Turns start at 1 (not 0)
- First turn: t=1, uses `act()` method
- Subsequent turns: t>=2, use `act_based_on_belief()`
- All data structures use 1-based turns
- Clarify in turn flow section

### 33. Agent Name Hardcoding
**Issue**: Original hardcodes agent names as "John" (actor) and "Jane" (audience). Plan should make configurable.

**Recommendation**:
- Add CLI arguments: `--actor_name` (default "John"), `--audience_name` (default "Jane")
- Or make configurable via config file
- Document in entity creation section

### 34. LLM Response Parsing Edge Cases
**Issue**: Plan mentions regex parsing but doesn't cover all edge cases.

**Recommendation**:
- Document: Handle multi-line responses
- Handle responses with text before/after number
- Handle scientific notation (though regex doesn't match it)
- Handle negative numbers (clamp to [0,1])
- Show robust parsing function

### 35. Memory State Persistence
**Issue**: Plan mentions checkpointing but doesn't detail what needs to be serialized.

**Recommendation**:
- Document: All memory state must be serializable
- PF particles and weights (lists of floats)
- PF history (list of dicts)
- Conversation, PE history, reflections (lists of dataclasses)
- Use `asdict()` for dataclasses, standard JSON for others
- Show serialization/deserialization logic

### 36. Component Initialization Order
**Issue**: Plan doesn't specify the order in which components should be initialized.

**Recommendation**:
- Document: Memory component first (others depend on it)
- Then CulturalNormsComponent and PersonalityTraitsComponent
- Then evaluation/particle filter components
- Finally act component (depends on all others)
- Show initialization sequence in prefab sections

### 37. Prompt Construction Pattern
**Issue**: Plan mentions prompts but doesn't show the pattern for combining elements.

**Recommendation**:
- Document pattern: `_prompt_header() + context + goal + instruction + data`
- Show how components contribute to prompt
- Document prompt length considerations
- Show example full prompt structure

### 38. Observation Context
**Issue**: Plan mentions observations but doesn't detail what context is included.

**Recommendation**:
- Document: Observation includes turn number
- Includes speaker name
- Includes full utterance (text + body)
- May include recent conversation history
- Show observation construction in game master

### 39. Error Recovery
**Issue**: Plan mentions fallbacks but doesn't detail recovery strategies.

**Recommendation**:
- Document: If LLM fails, retry with exponential backoff
- If parsing fails, use default values (0.5 for measurements, empty string for text)
- If component access fails, fall back to raw_log parsing
- If file save fails, print error but continue
- Show error handling in each component

### 40. EvaluationRecord Data Class
**Issue**: I_t needs to be stored in audience memory. Plan should add a new data class.

**Recommendation**:
- Add `EvaluationRecord` dataclass: `turn: int, I_t: float, utterance: Utterance`
- Add to IMPEMemoryComponent: `evaluation_history: List[EvaluationRecord]`
- Add method: `add_evaluation_record(turn, I_t, utterance)`
- IMPEAudienceEvaluationComponent stores I_t via this method
- Data extraction accesses I_t from evaluation_history

### 41. Modular Structure Requirements
**Issue**: Plan doesn't specify how to achieve modularity.

**Recommendation**:
- **Separate Constants File**: `examples/impression_management_constants.py`
  - All cultural norms (17 items)
  - All personality traits (11 items)
  - Default interview role
  - Default agent names
  - Default parameters (num_particles, sigmas, etc.)
- **Separate Utils Module**: `examples/impression_management_utils.py`
  - `parse_index_list()`
  - `select_by_indices()`
  - `generate_trait_scores()`
  - `create_output_directory()`
  - `format_conversation()`
  - `parse_llm_response()` (for dialogue/body extraction)
  - `extract_numeric_from_response()` (for I_t and measurements)
- **Separate Plotting Module**: `examples/impression_management_plotting.py`
  - `plot_learning_dynamics()`
  - Plot styling constants
- **Main Script**: `examples/impression_management_pe_conversation.py`
  - CLI argument parsing
  - Entity/prefab creation
  - Simulation orchestration
  - Data extraction
  - Pretty printing
- **Component File**: `concordia/components/agent/impression_management_pe.py`
  - All components
  - ParticleFilter class
  - Data classes
- **Prefab Files**: Separate files for each prefab type
- Document this structure in "File Organization Summary"

## Recommendations for Plan Updates

1. **Add Section**: "Utility Functions and Helpers" detailing all utility functions and their locations
2. **Add Section**: "Constants File Structure" specifying `impression_management_constants.py` with all constants
3. **Add Section**: "Modular File Organization" showing separation of concerns
4. **Add Section**: "Error Handling and Fallbacks" documenting all error cases and recovery
5. **Add Section**: "CLI Arguments Reference" with complete list and defaults
6. **Expand Section**: "Data Extraction" to show exact access patterns and edge cases
7. **Add Section**: "Local LLM Support" for Ollama integration
8. **Add Section**: "Prompt Construction Patterns" showing how prompts are built
9. **Clarify**: I_t storage mechanism in audience memory (add evaluation_history)
10. **Clarify**: PE computation (signed in PERecord, absolute in TurnLog)
11. **Add**: Exact regex patterns for all parsing operations
12. **Add**: Dependencies list with versions
13. **Add**: Component initialization order
14. **Add**: Turn numbering convention (1-based)
15. **Add**: Agent name configuration
16. **Add**: Observation context details
17. **Add**: Memory serialization requirements
18. **Add**: Plot generation details
19. **Add**: Conversation formatting method
20. **Add**: Context-based name switching logic

## Modular Structure Requirements

### Constants File: `examples/impression_management_constants.py`

**Purpose**: Centralize all constants, defaults, and configuration data.

**Contents**:
```python
# Cultural Norms (17 items)
ALL_CULTURAL_NORMS: List[CulturalNorm] = [
    CulturalNorm("Stated purpose first", "..."),
    # ... all 17 norms
]

# Personality Traits (11 items)
ALL_PERSONALITY_TRAITS: List[PersonalityTrait] = [
    PersonalityTrait("Detail-focused", "..."),
    # ... all 11 traits
]

# Default Interview Role
DEFAULT_INTERVIEW_ROLE = """
Role: Product Manager
...
"""

# Default Agent Names
DEFAULT_ACTOR_NAME = "John"
DEFAULT_AUDIENCE_NAME = "Jane"

# Default Parameters
DEFAULT_NUM_PARTICLES = 200
DEFAULT_PROCESS_SIGMA = 0.03
DEFAULT_OBS_SIGMA = 0.08
DEFAULT_RECENT_K = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_TURNS = 2
DEFAULT_SEED = 7

# Trait Score Ranges
AUDIENCE_TRAIT_SCORE_MIN = 2
AUDIENCE_TRAIT_SCORE_MAX = 3
ACTOR_TRAIT_SCORE_MIN = 0
ACTOR_TRAIT_SCORE_MAX = 1
```

### Utils Module: `examples/impression_management_utils.py`

**Purpose**: Reusable utility functions for parsing, formatting, and data manipulation.

**Functions**:
- `parse_index_list(s: str) -> List[int]`
- `select_by_indices(full_list, indices) -> List`
- `generate_trait_scores(rng, trait_list, is_audience) -> Dict[str, int]`
- `create_output_directory(save_dir: Optional[str]) -> str`
- `format_conversation(utterances: List[Utterance]) -> str`
- `parse_dialogue_and_body(response: str) -> Tuple[str, str]`
- `extract_numeric_from_response(response: str, default: float = 0.5) -> float`
- `clamp_to_range(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float`

### Plotting Module: `examples/impression_management_plotting.py`

**Purpose**: Visualization functions for analysis.

**Functions**:
- `plot_learning_dynamics(log: List[TurnLog], save_dir: str) -> None`
- `plot_pe_trajectory(turns, pe_values, save_path) -> None`
- `plot_belief_trajectories(turns, I_t, I_hat, save_path) -> None`
- `plot_learning_gain(turns, learning_gain, save_path) -> None`

### Data Classes Module: `concordia/components/agent/impression_management_pe.py`

**Purpose**: Shared data structures.

**Classes**:
- `Goal`
- `Utterance`
- `PERecord`
- `ReflectionRecord`
- `CulturalNorm`
- `PersonalityTrait`
- `EvaluationRecord` (NEW - for storing I_t)

### Component Module: `concordia/components/agent/impression_management_pe.py`

**Purpose**: All Concordia components.

**Classes**:
- `ParticleFilter` (utility class)
- `IMPEMemoryComponent`
- `IMPEAudienceEvaluationComponent`
- `IMPEActorParticleFilterComponent`
- `IMPEReflectionComponent`
- `IMPEActComponent`
- `CulturalNormsComponent`
- `PersonalityTraitsComponent`

### Prefab Modules: Separate files

- `concordia/prefabs/entity/impression_management_actor.py`
- `concordia/prefabs/entity/impression_management_audience.py`
- `concordia/prefabs/game_master/impression_management_pe.py`

### Main Script: `examples/impression_management_pe_conversation.py`

**Purpose**: Orchestration and CLI.

**Sections**:
1. Imports (from constants, utils, plotting, components, prefabs)
2. CLI argument parsing (all flags from original)
3. Configuration setup (load constants, generate trait scores)
4. Language model setup (OpenAI or local, with retry logic)
5. Entity/prefab creation
6. Simulation setup and execution
7. Data extraction (from components or raw_log)
8. Pretty printing (formatted console output)
9. Plotting (if matplotlib available)
10. JSON saving (to output directory)

### Additional Requirements

**Constants File Must Include**:
- All 17 cultural norms (name + description)
- All 11 personality traits (name + assertion)
- Default interview role (Product Manager)
- All default parameter values
- Trait score ranges
- Agent name defaults

**Utils Module Must Include**:
- All parsing functions (with error handling)
- All formatting functions
- Directory creation logic
- Numeric extraction with clamping
- Index parsing and selection

**Component Dependencies**:
- IMPEMemoryComponent must be initialized first
- CulturalNormsComponent and PersonalityTraitsComponent can be initialized in parallel
- IMPEAudienceEvaluationComponent depends on memory + norms + traits
- IMPEActorParticleFilterComponent depends on memory
- IMPEReflectionComponent depends on memory
- IMPEActComponent depends on memory + all above

**Data Flow Requirements**:
- I_t must be stored in audience's evaluation_history
- Actor's PF state must persist across turns
- All utterances must include body language
- PE must be computed both signed (in PERecord) and absolute (in TurnLog)
- Turn numbering must be 1-based throughout

## Updated File Organization

```
concordia/
├── components/
│   └── agent/
│       ├── impression_management_pe.py          # Components + ParticleFilter + Data classes
│       └── impression_management_pe_test.py
├── prefabs/
│   ├── entity/
│   │   ├── impression_management_actor.py
│   │   ├── impression_management_audience.py
│   │   ├── impression_management_actor_test.py
│   │   └── impression_management_audience_test.py
│   └── game_master/
│       └── impression_management_pe.py
examples/
├── impression_management_pe_conversation.py     # Main script
├── impression_management_constants.py            # All constants
├── impression_management_utils.py               # Utility functions
└── impression_management_plotting.py            # Plotting functions
docs/
└── impression_management_pe_components.md
```

## Priority Items

**High Priority** (Critical for implementation):
- I_t storage mechanism (add EvaluationRecord to memory)
- Utility functions location and implementation (separate utils module)
- CLI arguments completeness (all flags from original)
- Error handling patterns (retry logic, fallbacks)
- Observation format details (exact string format)
- Constants file structure (separate file with all constants)
- Modular file organization (separate concerns)

**Medium Priority** (Important for correctness):
- PE computation clarification (signed vs absolute)
- First turn handling details (I_hat_prev = None logic)
- Cultural norms initialization (one-time setup prompt)
- Prompt header formatting (exact format specification)
- PF history access patterns (indexing logic)
- Component initialization order (dependency sequence)
- Conversation formatting method (format_conversation)
- Context-based name switching (partner vs interviewee)
- Turn numbering convention (1-based)
- Memory serialization requirements (what to serialize)

**Low Priority** (Nice to have):
- Pretty print format (debugging aid)
- Local LLM support (Ollama integration)
- Interview role configuration (make configurable)
- Agent name configuration (make configurable)
- Plot generation details (exact matplotlib code)
- Observation context details (what's included)

## Summary: Critical Additions Needed

To fully replicate the original script with modular structure:

### 1. New Data Structure
- **EvaluationRecord**: Store I_t in audience memory
  ```python
  @dataclass
  class EvaluationRecord:
      turn: int
      I_t: float
      utterance: Utterance
  ```

### 2. Constants File (`examples/impression_management_constants.py`)
- All 17 cultural norms
- All 11 personality traits
- Default interview role
- All default parameters
- Trait score ranges
- Agent name defaults

### 3. Utils Module (`examples/impression_management_utils.py`)
- `parse_index_list()`
- `select_by_indices()`
- `generate_trait_scores()`
- `create_output_directory()` (with timestamp logic)
- `format_conversation()`
- `parse_dialogue_and_body()` (with regex and fallbacks)
- `extract_numeric_from_response()` (with regex and fallback)
- `clamp_to_range()`

### 4. Plotting Module (`examples/impression_management_plotting.py`)
- `plot_learning_dynamics()` (three plots)
- Individual plot functions for modularity

### 5. Component Enhancements
- **IMPEMemoryComponent**: Add `evaluation_history` and `add_evaluation_record()`
- **IMPEMemoryComponent**: Add `format_conversation()` method
- **IMPEAudienceEvaluationComponent**: Store I_t in evaluation_history
- **All Components**: Handle context-based name switching (partner vs interviewee)
- **IMPEActComponent**: Handle first turn (no I_hat history)

### 6. CLI Arguments (Complete List)
- `--turns` (default: 2)
- `--model` (default: "gpt-4o-mini")
- `--temperature` (default: 0.2)
- `--top_p` (default: 0.9)
- `--window` (default: 3)
- `--outfile` (default: "pe_conversation_log.json")
- `--no_audience_norms` (flag)
- `--no_traits` (flag)
- `--no_context` (flag)
- `--seed` (default: 7)
- `--save_dir` (default: None, creates timestamped dir)
- `--actor_name` (default: "John")
- `--audience_name` (default: "Jane")
- `--llm_type` (openai/local, default: openai)
- `--local_model` (for Ollama, default: "llama3.1:8b")

### 7. Error Handling
- LLM retry logic (exponential backoff, max 3 retries)
- Parsing fallbacks (default values)
- File I/O error handling
- Component access fallbacks (raw_log parsing)

### 8. Implementation Details to Add
- Exact regex patterns for all parsing
- Prompt construction patterns
- Observation format strings
- Turn numbering (1-based)
- PE computation (signed vs absolute)
- First turn special handling
- Component initialization order
- Memory serialization requirements

### 9. Documentation Requirements
- All constants documented
- All utility functions documented
- All CLI arguments documented
- Error handling documented
- Data flow diagrams
- Component dependency graph

## Verification Checklist

Before implementation, verify the plan includes:
- [ ] All 41 identified issues addressed
- [ ] Constants file structure specified
- [ ] Utils module structure specified
- [ ] Plotting module structure specified
- [ ] EvaluationRecord data class added
- [ ] I_t storage mechanism clarified
- [ ] All CLI arguments listed
- [ ] All utility functions specified
- [ ] All regex patterns documented
- [ ] Error handling patterns documented
- [ ] Component initialization order specified
- [ ] Modular file organization complete
- [ ] Dependencies listed
- [ ] Turn numbering convention specified
- [ ] PE computation clarified (signed vs absolute)
