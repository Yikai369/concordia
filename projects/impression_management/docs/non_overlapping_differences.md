# Non-Overlapping Differences: Unique to Each Comparison

## Overview

This document identifies differences that appear in **ONLY ONE** of the two comparison documents:
1. Differences **ONLY** in `pe_conversation_openai.py` vs `pe_conversation_prototype.py`
2. Differences **ONLY** in `impression_management_standard` vs `pe_conversation_prototype.py`

These represent features that differ between the two standards themselves, or are unique to the framework-based standard.

---

## Differences ONLY in `impression_management_standard` vs Prototype

### 1. Framework Architecture

**`impression_management_standard`:**
- ✅ Uses **Concordia framework**
- ✅ Component-based architecture
- ✅ Automatic component lifecycle
- ✅ Framework-managed execution

**`pe_conversation_prototype.py`:**
- ❌ Standalone Python script
- ❌ Direct `Agent` class methods
- ❌ Manual component coordination
- ❌ No framework dependencies

**`pe_conversation_openai.py`:**
- ❌ Also standalone (same as prototype)
- ❌ Also uses direct `Agent` class methods
- ❌ Also manual coordination

**Why Non-Overlapping:** `pe_conversation_openai.py` is also standalone, so this difference only appears when comparing `impression_management_standard` (framework-based) vs prototype.

---

### 2. Execution Model

**`impression_management_standard`:**
```python
# Uses Concordia's standard simulation loop
sim = simulation.Simulation(config=config, model=model, embedder=embedder)
results_log = sim.play(max_steps=args.turns * 2, raw_log=raw_log)
turn_logs = extract_turn_data_from_entities(sim, agent_a_name, agent_b_name, args.turns)
```

**`pe_conversation_prototype.py`:**
```python
# Manual 4-step conversation loop
for t in range(1, self.total_turns + 1):
    actor_utt = actor.act_based_on_belief(...)
    I_t, audience_reply = audience.audience_evaluate_and_respond(...)
    I_hat, ess = actor.actor_update_particles(...)
    refl = actor.learning(turn=t)
```

**`pe_conversation_openai.py`:**
```python
# Also uses manual loop (same as prototype)
for t in range(1, self.total_turns + 1):
    speaker_utt = speaker.act_based_on_belief(...)
    I_t, listener_reply = listener.audience_evaluate_and_respond(...)
    I_hat, ess = speaker.actor_update_particles(...)
    refl = speaker.learning(turn=t)
```

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and prototype use manual loops, so this difference only appears when comparing `impression_management_standard` (uses `sim.play()`) vs prototype.

---

### 3. Component System

**`impression_management_standard`:**
- ✅ Component-based architecture
- ✅ `IMPEActComponent`, `IMPEAudienceComponent`, `IMPEMemoryComponent`, etc.
- ✅ Components automatically invoked via `entity.observe()` and `entity.act()`
- ✅ Component lifecycle methods (`pre_observe()`, `post_act()`, etc.)

**`pe_conversation_prototype.py`:**
- ❌ Direct `Agent` class methods
- ❌ Manual method calls
- ❌ No component abstraction

**`pe_conversation_openai.py`:**
- ❌ Also uses direct `Agent` class methods
- ❌ Also manual method calls
- ❌ Also no component abstraction

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and prototype use direct methods, so this difference only appears when comparing `impression_management_standard` (component-based) vs prototype.

---

### 4. Modularity & File Structure

**`impression_management_standard`:**
- ✅ Multiple files (config, setup, entities, results, etc.)
- ✅ Modular component files
- ✅ Separate constants file
- ✅ Organized project structure

**`pe_conversation_prototype.py`:**
- ❌ Single file (~1185 lines)
- ❌ All code in one file
- ❌ No modular separation

**`pe_conversation_openai.py`:**
- ❌ Also single file
- ❌ Also all code in one file
- ❌ Also no modular separation

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and prototype are single-file scripts, so this difference only appears when comparing `impression_management_standard` (modular) vs prototype.

---

### 5. Configurability

**`impression_management_standard`:**
- ✅ Configurable agent names via `--actor_name` and `--audience_name`
- ✅ Configurable interview role via `Goal.role` (stored in constants)
- ✅ Configurable via command-line arguments
- ✅ Can disable features via flags (`--no_context`, `--no_traits`, etc.)

**`pe_conversation_prototype.py`:**
- ❌ Hardcoded agent names ("Riffer", "Caden")
- ❌ Hardcoded interview role ("Customer Service Agent")
- ❌ Hardcoded file paths
- ❌ Hardcoded interview questions/experiences

**`pe_conversation_openai.py`:**
- ⚠️ Also has hardcoded names ("John", "Jane")
- ⚠️ Also has hardcoded role ("Product Manager")
- ⚠️ But role is defined in `main()`, not deeply embedded

**Why Non-Overlapping:** While `pe_conversation_openai.py` also has some hardcoded values, `impression_management_standard` has much more configurability, making this a more significant difference in that comparison.

---

### 6. Self-Assessment Component

**`impression_management_standard`:**
- ✅ Optional `IMPESelfAssessmentComponent`
- ✅ Threshold-based consistency checking (default: 0.7)
- ✅ Can enable/disable via `--enable_self_assessment`
- ✅ Can disable revision while keeping assessment via `--disable_revision`
- ✅ Provides feedback on inconsistencies
- ✅ Logs consistency scores

**`pe_conversation_prototype.py`:**
- ❌ Unconditional self-reflection (always revises if traits exist)
- ❌ No consistency score
- ❌ No threshold
- ❌ No feedback mechanism

**`pe_conversation_openai.py`:**
- ❌ No self-reflection on responses
- ❌ Only basic reflection in `learning()` method
- ❌ No consistency checking

**Why Non-Overlapping:** `pe_conversation_openai.py` doesn't have self-reflection at all, while `impression_management_standard` has a sophisticated optional component. The prototype's unconditional approach is different from both.

---

### 7. Error Handling

**`impression_management_standard`:**
- ✅ Framework-managed error handling
- ✅ Component-level error recovery
- ✅ Automatic retry mechanisms (if framework provides)

**`pe_conversation_prototype.py`:**
- ⚠️ Manual error handling
- ⚠️ Explicit retry logic with exponential backoff

**`pe_conversation_openai.py`:**
- ✅ Also has explicit retry logic with exponential backoff (same as prototype)

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and prototype have similar manual error handling, so this difference only appears when comparing `impression_management_standard` (framework-managed) vs prototype.

---

### 8. Plotting

**`impression_management_standard`:**
- ✅ Plots enabled by default
- ✅ Can disable with `--no_plots` flag
- ✅ Generates: `pe.png`, `delta_I.png`, `learning_gain.png`

**`pe_conversation_prototype.py`:**
- ❌ Plotting commented out (line 1170)
- ❌ Same plotting function exists but not called

**`pe_conversation_openai.py`:**
- ✅ Plots enabled by default
- ✅ Always generates plots
- ✅ Same three plots

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and `impression_management_standard` have plotting enabled, so this difference only appears when comparing `impression_management_standard` vs prototype (where it's disabled).

---

## Differences ONLY in `pe_conversation_openai.py` vs Prototype

**Note:** The `reflection_text` field difference was initially listed here but is actually an **overlapping difference** (both standards have it, prototype doesn't).

### 1. Cultural Norm Initialization

**`pe_conversation_openai.py`:**
```python
def initialize_cultural_norms(self, norms: List[CulturalNorm]) -> None:
    """Set cultural norms for the agent."""
    prompt = f"""You are {self.name}. You are in an alternative world in the year 3025...
        If you fail to do so, you will be unsuccessful..."""
    self.llm(prompt)
```

**`pe_conversation_prototype.py`:**
- ❌ No separate `initialize_cultural_norms()` method
- ✅ Norms included directly in `_prompt_header()`

**`impression_management_standard`:**
- ✅ Uses `CulturalNormsComponent`
- ✅ Automatic component initialization

**Why Non-Overlapping:** `impression_management_standard` uses components, so this method-level difference only appears when comparing `pe_conversation_openai.py` (has method) vs prototype (no method).

---

### 2. Trait Score Generation

**`pe_conversation_openai.py`:**
```python
# Explicitly generates scores
aud_trait_scores = generate_trait_scores(rng, traits, is_audience=True)  # Scores 2-3
actor_trait_scores = generate_trait_scores(rng, traits, is_audience=False)  # Scores 0-1

# Passes scores to agent
agent = Agent(
    traits=traits,
    trait_scores=actor_trait_scores,  # ← Scores provided
)
```

**`pe_conversation_prototype.py`:**
```python
# No score generation
agent = Agent(
    traits=aud_traits,
    trait_scores=None,  # ← Always None
)
```

**`impression_management_standard`:**
- ✅ Also uses `generate_trait_scores()`
- ✅ Also passes scores to components
- ✅ Similar to `pe_conversation_openai.py`

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and `impression_management_standard` use score generation, so this difference only appears when comparing `pe_conversation_openai.py` vs prototype.

---

### 3. Cultural Norms for Actor

**`pe_conversation_openai.py`:**
```python
A = Agent(  # Actor
    name="John",
    cultural_norms=[],  # ← Actor has NO norms
    ...
)

B = Agent(  # Audience
    name="Jane",
    cultural_norms=aud_norms,  # ← Audience has norms
    ...
)
```

**`pe_conversation_prototype.py`:**
```python
A = Agent(  # Actor (Riffer)
    name="Riffer",
    cultural_norms=aud_norms,  # ← Actor ALSO has norms!
    ...
)

B = Agent(  # Audience (Caden)
    name="Caden",
    cultural_norms=aud_norms,  # ← Audience has norms
    ...
)
```

**`impression_management_standard`:**
- ✅ Also gives norms only to audience
- ✅ Actor has no norms (similar to `pe_conversation_openai.py`)

**Why Non-Overlapping:** Both `pe_conversation_openai.py` and `impression_management_standard` give norms only to audience, so this difference only appears when comparing `pe_conversation_openai.py` vs prototype.

---

### 4. Reflection Text in TurnLog

**`pe_conversation_openai.py`:**
```python
@dataclass
class TurnLog:
    ...
    reflection_text: str  # ← Has reflection text
    ...
```

**`pe_conversation_prototype.py`:**
```python
@dataclass
class TurnLog:
    ...
    # Note: No reflection_text field
    ...
```

**`impression_management_standard`:**
- ✅ Also has `reflection_text` in TurnLog (line 20 in `models.py`)
- ✅ Similar to `pe_conversation_openai.py`

**Correction:** This is actually an **OVERLAPPING difference**, not a non-overlapping one. Both standards have `reflection_text`, while prototype does not. This should be moved to the overlapping differences document.

---

## Summary Table

| Difference | `pe_conversation_openai.py` | `impression_management_standard` | `pe_conversation_prototype.py` | Appears In |
|------------|----------------------------|----------------------------------|--------------------------------|------------|
| **Framework** | Standalone | ✅ Concordia | Standalone | Only standard vs prototype |
| **Execution** | Manual loop | ✅ `sim.play()` | Manual loop | Only standard vs prototype |
| **Components** | Direct methods | ✅ Component-based | Direct methods | Only standard vs prototype |
| **Modularity** | Single file | ✅ Multi-file | Single file | Only standard vs prototype |
| **Configurability** | Some hardcoded | ✅ Highly configurable | Hardcoded | Only standard vs prototype |
| **Self-Assessment** | None | ✅ Optional component | Unconditional | Only standard vs prototype |
| **Plotting** | ✅ Enabled | ✅ Enabled | ❌ Disabled | Only standard vs prototype |
| **Cultural Norm Init** | ✅ Has method | Component-based | ❌ No method | Only openai vs prototype |
| **Trait Scores** | ✅ Generated | ✅ Generated | ❌ None | Only openai vs prototype |
| **Actor Norms** | ❌ None | ❌ None | ✅ Has norms | Only openai vs prototype |
| **Reflection Text** | ✅ In TurnLog | ✅ In TurnLog | ❌ Not in TurnLog | **OVERLAPPING** (both standards have it) |

---

## Key Insights

### Framework vs Standalone Differences
- **Only in `impression_management_standard` vs prototype:**
  - Framework integration
  - Component architecture
  - Execution model
  - Modularity
  - Configurability
  - Self-assessment component

These represent the **architectural differences** between framework-based and standalone implementations.

### Implementation Detail Differences
- **Only in `pe_conversation_openai.py` vs prototype:**
  - Cultural norm initialization method
  - Trait score generation usage
  - Actor cultural norms assignment
  - Reflection text in TurnLog

These represent **implementation detail differences** between two standalone scripts that share the same architecture.

---

**Document Version:** 1.0
**Last Updated:** 2025-12-27
