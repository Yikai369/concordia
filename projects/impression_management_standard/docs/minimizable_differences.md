# Minimizable Differences Between Actor and Audience

This document identifies differences between Actor and Audience agent implementations that can be minimized or unified to reduce code duplication and improve maintainability.

## Current Differences

### 1. **Component Assembly Code Duplication** ⚠️ **HIGH PRIORITY**

**Current State:**
- Both prefabs have nearly identical component assembly code (lines 280-320 in actor, 247-286 in audience)
- Only differences are:
  - Actor: Has `IMPEActorParticleFilterComponent` and `IMPEReflectionComponent` at step 7
  - Audience: Has `IMPEAudienceEvaluationComponent` at step 7
  - Otherwise, the ordering and structure are identical

**Impact:**
- ~40 lines of duplicated code
- Changes to component ordering must be made in two places
- Risk of inconsistencies if one prefab is updated but not the other

**Minimization Strategy:**
```python
# Create shared helper function
def assemble_impe_components(
    instructions_comp,
    self_perception_comp,
    situation_perception_comp,
    person_by_situation_comp,
    memory,
    impe_memory,
    observation_to_memory,
    world_context_comp=None,
    cultural_norms_comp=None,
    personality_traits_comp=None,
    # Actor-specific
    actor_pf=None,
    reflection=None,
    # Audience-specific
    audience_eval=None,
):
    components = {}

    # Common components (1-6)
    if instructions_comp:
        components['Instructions'] = instructions_comp
    if self_perception_comp:
        components['SelfPerception'] = self_perception_comp
    if situation_perception_comp:
        components['SituationPerception'] = situation_perception_comp
    if person_by_situation_comp:
        components['PersonBySituation'] = person_by_situation_comp
    components['Memory'] = memory
    components['IMPE_Memory'] = impe_memory
    components['ObservationToMemory'] = observation_to_memory

    # Role-specific components (7)
    if actor_pf:
        components['IMPE_ActorParticleFilter'] = actor_pf
    if reflection:
        components['IMPE_Reflection'] = reflection
    if audience_eval:
        components['IMPE_AudienceEvaluation'] = audience_eval

    # Common optional components (8-10)
    if world_context_comp:
        components['WorldContext'] = world_context_comp
    if cultural_norms_comp:
        components['CulturalNorms'] = cultural_norms_comp
    if personality_traits_comp:
        components['PersonalityTraits'] = personality_traits_comp

    return components
```

**Benefits:**
- Single source of truth for component ordering
- Easier to maintain and update
- Reduces risk of inconsistencies

---

### 2. **Cultural Norms Initialization Inconsistency** ⚠️ **MEDIUM PRIORITY**

**Current State:**
- **Audience**: Calls `cultural_norms_comp.initialize_norms(model, entity_name)` after creating the component (line 170)
- **Actor**: Does NOT call `initialize_norms()` at all

**Impact:**
- Inconsistent behavior: audience gets one-time initialization prompt, actor doesn't
- If actor ever needs norms, initialization would be missing
- The initialization is already handled in `get_norms_text()` when agent_name is provided, so this may be redundant

**Minimization Strategy:**
- **Option A**: Remove explicit initialization from audience (rely on `get_norms_text()` with agent_name)
- **Option B**: Add initialization to actor if norms are provided
- **Option C**: Make initialization conditional based on a parameter

**Recommendation:** Option A - The `get_norms_text(agent_name)` method already includes the full initialization context in every prompt, making the one-time initialization redundant. Remove it from audience prefab.

---

### 3. **Act Component Wrapper Pattern** ✅ **ALREADY UNIFIED**

**Current State:**
- Both use the same pattern for optional self-assessment wrapper
- Code is identical (lines 258-278 in actor, 225-245 in audience)

**Status:** This is already well-unified. No changes needed.

---

### 4. **Parameter Defaults and Structure** ⚠️ **LOW PRIORITY**

**Current State:**
- Both prefabs have similar `params` dictionaries
- Some defaults differ (e.g., `enable_self_assessment` defaults to `False` for both, which is fine)
- Structure is nearly identical

**Impact:**
- Minor duplication in parameter definitions
- Could be unified but low priority since defaults are configuration-specific

**Minimization Strategy:**
- Create shared base class or mixin for common parameters
- Or document that parameters should be kept in sync

**Recommendation:** Low priority - the duplication is minimal and parameters may diverge for good reasons.

---

### 5. **Component Creation Logic** ⚠️ **MEDIUM PRIORITY**

**Current State:**
- Both prefabs have identical logic for creating:
  - Instructions component
  - SelfPerception component
  - SituationPerception component
  - PersonBySituation component
  - IMPEMemoryComponent
  - WorldContextComponent
  - PersonalityTraitsComponent
  - Self-assessment wrapper

**Impact:**
- ~150 lines of duplicated code across both prefabs
- Changes to component creation must be made in two places

**Minimization Strategy:**
```python
# Create shared helper functions
def create_identity_components(
    entity_name: str,
    model: LanguageModel,
    enable_instructions: bool,
    enable_self_perception: bool,
    enable_situation_perception: bool,
    enable_person_by_situation: bool,
) -> dict:
    """Create Instructions, SelfPerception, SituationPerception, PersonBySituation."""
    components = {}
    keys = {}

    if enable_instructions:
        keys['instructions'] = 'Instructions'
        components['instructions'] = instructions.Instructions(
            agent_name=entity_name,
            pre_act_label='\nRole playing instructions',
        )

    if enable_self_perception:
        keys['self_perception'] = 'SelfPerception'
        components['self_perception'] = question_of_recent_memories.SelfPerception(
            model=model,
            pre_act_label=f'\nQuestion: What kind of person is {entity_name}?\nAnswer',
        )

    if enable_situation_perception:
        keys['situation_perception'] = 'SituationPerception'
        components['situation_perception'] = question_of_recent_memories.SituationPerception(
            model=model,
            pre_act_label=f'\nQuestion: What kind of situation is {entity_name} in right now?\nAnswer',
        )

    if enable_person_by_situation and keys.get('self_perception') and keys.get('situation_perception'):
        keys['person_by_situation'] = 'PersonBySituation'
        components['person_by_situation'] = question_of_recent_memories.PersonBySituation(
            model=model,
            components=[keys['self_perception'], keys['situation_perception']],
            pre_act_label=f'\nQuestion: What would a person like {entity_name} do in a situation like this?\nAnswer',
        )
    elif enable_person_by_situation:
        import warnings
        warnings.warn(
            f"PersonBySituation requires both SelfPerception and SituationPerception. "
            f"Disabling PersonBySituation for {entity_name}.",
            UserWarning
        )

    return components, keys

def create_impe_memory_component(goal, recent_k):
    """Create IMPEMemoryComponent."""
    return impe_components.IMPEMemoryComponent(
        goal=goal,
        recent_k=recent_k,
        pre_act_label='\nIMPE Memory',
    )

def create_world_context_component(enable_world_building, enable_interview_context):
    """Create WorldContextComponent."""
    if enable_world_building or enable_interview_context:
        return impe_components.WorldContextComponent(
            enable_world_building=enable_world_building,
            enable_interview_context=enable_interview_context,
            pre_act_label='\nWorld Context',
        )
    return None

# Similar helpers for other components...
```

**Benefits:**
- Reduces code duplication significantly
- Ensures consistent component creation
- Easier to test and maintain

---

### 6. **Memory Component Creation** ✅ **ALREADY UNIFIED**

**Current State:**
- Both create `AssociativeMemory` and `ObservationToMemory` identically
- Code is the same

**Status:** Already unified. No changes needed.

---

## Summary of Minimizable Differences

| Difference | Priority | Impact | Effort | Recommendation |
|------------|----------|--------|--------|----------------|
| Component Assembly Code | **HIGH** | High duplication, maintenance risk | Medium | Create shared helper function |
| Cultural Norms Initialization | **MEDIUM** | Inconsistent behavior | Low | Remove redundant initialization |
| Component Creation Logic | **MEDIUM** | High duplication | High | Create shared helper functions |
| Act Component Wrapper | ✅ | Already unified | - | No action needed |
| Parameter Defaults | **LOW** | Minor duplication | Low | Document sync requirements |
| Memory Components | ✅ | Already unified | - | No action needed |

## Recommended Implementation Order

1. **Phase 1: Remove redundant cultural norms initialization** (Quick win)
   - Remove `initialize_norms()` call from audience prefab
   - Verify that `get_norms_text(agent_name)` provides sufficient context

2. **Phase 2: Create shared component assembly helper** (High impact)
   - Extract component assembly logic to shared utility
   - Update both prefabs to use the helper
   - Test thoroughly to ensure no regressions

3. **Phase 3: Create shared component creation helpers** (Optional, high effort)
   - Extract component creation logic to shared utilities
   - Update both prefabs to use helpers
   - Consider if the abstraction is worth the complexity

## Functional Differences (Should NOT be minimized)

These differences are intentional and should remain:

1. **Particle Filter Component**
   - Actor has `IMPEActorParticleFilterComponent` (tracks I_hat)
   - Audience does not (uses I_t directly)
   - **Reason**: Core functional difference - actor needs belief tracking

2. **Reflection Component**
   - Actor has `IMPEReflectionComponent` (learns how to improve)
   - Audience does not
   - **Reason**: Actor needs to adapt, audience just evaluates

3. **Evaluation Component**
   - Audience has `IMPEAudienceEvaluationComponent` (evaluates actor)
   - Actor does not
   - **Reason**: Core functional difference - audience evaluates, actor is evaluated

4. **Act Component Type**
   - Actor uses `IMPEActComponent` (generates based on I_hat)
   - Audience uses `SimpleAudienceActComponent` (returns stored response)
   - **Reason**: Different action generation strategies

5. **Cultural Norms Default**
   - Actor typically has no norms
   - Audience typically has norms
   - **Reason**: Configuration difference, not structural

## Conclusion

The main opportunities for minimization are:

1. **Code structure duplication** - Component assembly and creation logic
2. **Inconsistent initialization** - Cultural norms initialization pattern

The functional differences (particle filter, reflection, evaluation) are intentional and should remain as they represent core behavioral differences between actor and audience roles.




