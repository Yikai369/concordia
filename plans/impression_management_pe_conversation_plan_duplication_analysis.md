# Duplication Analysis: Impression Management PE vs Existing PE Components

## Overview

This document analyzes existing Concordia components and prefabs to identify what can be reused, extended, or must be created new for the impression management PE conversation system.

## Existing PE Conversation System

### Location
- **Components**: `concordia/components/agent/pe_conversation.py`
- **Prefab**: `concordia/prefabs/entity/pe_entity.py`
- **Example**: `examples/pe_conversation_concordia.py`

### Existing Components

#### 1. Data Classes (in `pe_conversation.py`)
- ✅ `Goal` - Has: `name`, `description`, `ideal: float = 1.0`
- ✅ `Utterance` - Has: `turn`, `speaker`, `text`
- ✅ `PERecord` - Has: `turn`, `partner_text`, `estimate`, `pe`
- ✅ `ReflectionRecord` - Has: `turn`, `text`

#### 2. Components
- ✅ `PEMemoryComponent` - Stores conversation, PE history, reflections, goal
- ✅ `PEEstimationComponent` - Estimates state and computes PE (simple: ideal - estimate)
- ✅ `PEReflectionComponent` - Generates reflections based on PE
- ✅ `PEActComponent` - Generates utterances based on PE history

#### 3. Prefab
- ✅ `pe_entity.Entity` - Prefab that creates entity with all PE components

## Comparison: Impression Management vs Existing PE

### Data Classes

| Class | Existing | Impression Management | Action |
|-------|----------|----------------------|--------|
| `Goal` | `name`, `description`, `ideal` | + `role: str` (interview context) | **EXTEND** - Add `role` field |
| `Utterance` | `turn`, `speaker`, `text` | + `body: str` (body language) | **EXTEND** - Add `body` field |
| `PERecord` | `turn`, `partner_text`, `estimate`, `pe` | Same structure, but PE computation differs | **REUSE** - Same structure |
| `ReflectionRecord` | `turn`, `text` | Same structure | **REUSE** - Same structure |
| `EvaluationRecord` | ❌ Doesn't exist | `turn`, `I_t`, `utterance` | **CREATE NEW** |
| `CulturalNorm` | ❌ Doesn't exist | `name`, `description` | **CREATE NEW** |
| `PersonalityTrait` | ❌ Doesn't exist | `name`, `assertion` | **CREATE NEW** |

### Components

| Component | Existing | Impression Management | Action |
|-----------|----------|----------------------|--------|
| Memory | `PEMemoryComponent` - conversation, PE history, reflections | + particle filter state, evaluation_history | **EXTEND** or **CREATE NEW** (`IMPEMemoryComponent`) |
| Estimation | `PEEstimationComponent` - simple LLM estimation | Audience: LLM evaluation (I_t), Actor: Particle filter (I_hat) | **CREATE NEW** - Different logic |
| Reflection | `PEReflectionComponent` - based on PE | Based on I_hat (belief), not PE | **CREATE NEW** - Different logic |
| Act | `PEActComponent` - based on PE history | Based on particle filter belief, handles first turn differently | **CREATE NEW** - Different logic |
| Particle Filter | ❌ Doesn't exist | Core algorithm for belief tracking | **CREATE NEW** - Utility class |
| Cultural Norms | ❌ Doesn't exist | Injects norms into prompts, initialization | **CREATE NEW** |
| Personality Traits | ❌ Doesn't exist | Injects traits with scores into prompts | **CREATE NEW** |

## Recommendations

### 1. Data Classes Strategy

#### Option A: Extend Existing (Recommended)
- **Extend `Goal`**: Add optional `role: str | None = None` field
- **Extend `Utterance`**: Add optional `body: str = ""` field
- **Reuse `PERecord` and `ReflectionRecord`**: Same structure
- **Create new**: `EvaluationRecord`, `CulturalNorm`, `PersonalityTrait`

**Pros**:
- Maintains compatibility with existing PE system
- Reuses proven data structures
- Minimal code duplication

**Cons**:
- Requires modifying existing code (but backward compatible if optional fields)

#### Option B: Create Separate Classes
- Create `IMPEGoal`, `IMPEUtterance`, etc. in impression_management_pe.py

**Pros**:
- No modification to existing code
- Complete separation

**Cons**:
- Code duplication
- Maintenance burden (two sets of similar classes)

**Recommendation**: **Option A** - Extend existing classes with optional fields.

### 2. Memory Component Strategy

#### Option A: Extend PEMemoryComponent (Recommended)
- Create `IMPEMemoryComponent` that inherits from `PEMemoryComponent`
- Add: `particle_filter_state`, `evaluation_history`, `format_conversation()`
- Override `get_state()` / `set_state()` to include new fields

**Pros**:
- Reuses existing conversation/PE/reflection storage logic
- Minimal duplication
- Can still use existing methods

**Cons**:
- Requires understanding inheritance
- Need to ensure backward compatibility

#### Option B: Create Independent Component
- Create `IMPEMemoryComponent` from scratch
- Copy relevant methods from `PEMemoryComponent`

**Pros**:
- Complete independence
- No risk of breaking existing code

**Cons**:
- Code duplication
- Maintenance burden

**Recommendation**: **Option A** - Inherit and extend.

### 3. Component Strategy

#### Estimation Component
- **Cannot reuse** `PEEstimationComponent` - logic is too different
  - Existing: Simple LLM estimation, PE = ideal - estimate
  - Impression Management:
    - Audience: LLM evaluation (I_t) with cultural norms/traits
    - Actor: Particle filter update, measurement extraction
- **Action**: Create `IMPEAudienceEvaluationComponent` and `IMPEActorParticleFilterComponent` as new components

#### Reflection Component
- **Cannot reuse** `PEReflectionComponent` - logic differs
  - Existing: Reflection based on PE value
  - Impression Management: Reflection based on I_hat (belief state)
- **Action**: Create `IMPEReflectionComponent` as new component

#### Act Component
- **Cannot reuse** `PEActComponent` - logic differs significantly
  - Existing: Simple PE-based utterance generation
  - Impression Management:
    - Particle filter belief-based
    - First turn special handling
    - Body language generation
    - Cultural norms/traits in prompts
- **Action**: Create `IMPEActComponent` as new component

### 4. Prefab Strategy

- **Cannot reuse** `pe_entity.Entity` - different component set
- **Action**: Create new prefabs:
  - `impression_management_actor.py` - Actor entity
  - `impression_management_audience.py` - Audience entity
  - `impression_management_pe.py` (game master) - Asymmetric conversation flow

### 5. Particle Filter

- **New utility class** - No existing equivalent
- **Action**: Create `ParticleFilter` class in `impression_management_pe.py`
- This is a pure utility class, no duplication concerns

### 6. Cultural Norms & Personality Traits

- **New components** - No existing equivalent
- **Action**: Create `CulturalNormsComponent` and `PersonalityTraitsComponent`
- These are domain-specific, no duplication concerns

## Implementation Plan Updates

### File Organization

```
concordia/
├── components/
│   └── agent/
│       ├── pe_conversation.py                    # EXISTING - extend Goal, Utterance
│       └── impression_management_pe.py           # NEW - all IMPE components
├── prefabs/
│   ├── entity/
│   │   ├── pe_entity.py                          # EXISTING - keep as-is
│   │   ├── impression_management_actor.py        # NEW
│   │   └── impression_management_audience.py     # NEW
│   └── game_master/
│       └── impression_management_pe.py           # NEW
```

### Code Changes Required

#### 1. Extend `pe_conversation.py` Data Classes

```python
@dataclass
class Goal:
  name: str
  description: str
  ideal: float = 1.0
  role: str | None = None  # NEW - for interview context

@dataclass
class Utterance:
  turn: int
  speaker: str
  text: str
  body: str = ""  # NEW - for body language
```

#### 2. Create `impression_management_pe.py`

- New data classes: `EvaluationRecord`, `CulturalNorm`, `PersonalityTrait`
- Utility class: `ParticleFilter`
- Components:
  - `IMPEMemoryComponent` (extends `PEMemoryComponent`)
  - `IMPEAudienceEvaluationComponent` (new)
  - `IMPEActorParticleFilterComponent` (new)
  - `IMPEReflectionComponent` (new)
  - `IMPEActComponent` (new)
  - `CulturalNormsComponent` (new)
  - `PersonalityTraitsComponent` (new)

#### 3. Import Strategy

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

## Summary

### Reuse/Extend
- ✅ `Goal` - Extend with `role` field
- ✅ `Utterance` - Extend with `body` field
- ✅ `PERecord` - Reuse as-is
- ✅ `ReflectionRecord` - Reuse as-is
- ✅ `PEMemoryComponent` - Extend for `IMPEMemoryComponent`

### Create New
- ❌ `EvaluationRecord` - New data class
- ❌ `CulturalNorm` - New data class
- ❌ `PersonalityTrait` - New data class
- ❌ `ParticleFilter` - New utility class
- ❌ `IMPEAudienceEvaluationComponent` - New component (different logic)
- ❌ `IMPEActorParticleFilterComponent` - New component (particle filter)
- ❌ `IMPEReflectionComponent` - New component (different logic)
- ❌ `IMPEActComponent` - New component (different logic)
- ❌ `CulturalNormsComponent` - New component
- ❌ `PersonalityTraitsComponent` - New component
- ❌ All prefabs - New (different component set)

### No Duplication
- All new components serve different purposes or use different algorithms
- Extending existing classes maintains compatibility
- Clear separation: existing PE system remains unchanged, impression management adds specialized components

## Action Items for Plan Update

1. ✅ Update plan to specify extending `Goal` and `Utterance` in `pe_conversation.py`
2. ✅ Update plan to specify `IMPEMemoryComponent` inheriting from `PEMemoryComponent`
3. ✅ Clarify that new components are necessary due to different algorithms/logic
4. ✅ Document import strategy (importing from `pe_conversation`)
5. ✅ Note that `PERecord` and `ReflectionRecord` are reused as-is
6. ✅ Specify that prefabs are new (cannot reuse `pe_entity`)
