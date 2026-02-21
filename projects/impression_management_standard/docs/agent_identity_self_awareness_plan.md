# Implementation Plan: Agent Identity and Self-Awareness Questions

## Overview

This plan implements standard Concordia framework identity and self-awareness components to improve agent self-awareness, consistency, and framework alignment. These components help agents develop a clear sense of identity and understand their current situation.

## Current State Analysis

### What Exists:
- `IMPEActComponent` - Generates actions based on belief
- `IMPEMemoryComponent` - Stores conversation history, reflections, particle filter state
- `CulturalNormsComponent` - Provides cultural norms context
- `PersonalityTraitsComponent` - Provides personality traits context
- Goal information is included in prompts

### What's Missing:
- **Instructions Component**: No role-playing instructions explaining experimental context
- **SelfPerception Component**: No explicit "who am I?" questions
- **SituationPerception Component**: No "what situation am I in?" questions
- **PersonBySituation Component**: No "what would a person like me do?" reasoning

## Design Decisions

### Component Selection Strategy

**Required Components:**
1. **Instructions Component** - Essential for framework alignment and experimental context

**Recommended Components:**
2. **SelfPerception Component** - Helps maintain consistent character identity
3. **SituationPerception Component** - Helps agents understand current context

**Optional Components:**
4. **PersonBySituation Component** - Can improve action quality but adds complexity

### Component Ordering

Components should be ordered to ensure dependencies are met:
1. **Instructions** (first - provides experimental context)
2. **Observation/ObservationToMemory** (if not already present)
3. **SelfPerception** (can use traits, norms, memories)
4. **SituationPerception** (uses observations and memories)
5. **PersonBySituation** (depends on SelfPerception and SituationPerception)
6. **Cultural Norms** (existing)
7. **Personality Traits** (existing)
8. **IMPE Memory** (existing)
9. **Other IMPE components** (existing)

## Implementation Plan

### Phase 1: Add Instructions Component

#### Step 1.1: Update Actor Prefab
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Add after imports (around line 20):**
```python
from concordia.components.agent import instructions
```

**Add in `build()` method, before IMPE Memory component (around line 96):**
```python
# Instructions component (role-playing context, optional)
enable_instructions = bool(
    self.params.get('enable_instructions', True)  # Default: enabled
)

instructions_key = None
instructions_comp = None
if enable_instructions:
    instructions_key = 'Instructions'
    instructions_comp = instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nRole playing instructions',
    )
```

**Add to components dictionary (around line 188), BEFORE existing components:**
```python
# Assemble components
components_of_agent = {}

# Add Instructions first if enabled
if instructions_key:
    components_of_agent[instructions_key] = instructions_comp

# Then add existing components
components_of_agent[memory_key] = memory
components_of_agent[impe_memory_key] = impe_memory
components_of_agent[actor_pf_key] = actor_pf
components_of_agent[reflection_key] = reflection
components_of_agent[observation_to_memory_key] = observation_to_memory
```

**Add to params default factory (around line 36):**
```python
'enable_instructions': True,
```

#### Step 1.2: Update Audience Prefab
**File:** `projects/impression_management_standard/simple_audience_prefab.py`

**Add after imports:**
```python
from concordia.components.agent import instructions
```

**Add in `build()` method, before IMPE Memory component:**
```python
# Instructions component (role-playing context, optional)
enable_instructions = bool(
    self.params.get('enable_instructions', True)  # Default: enabled
)

instructions_key = None
instructions_comp = None
if enable_instructions:
    instructions_key = 'Instructions'
    instructions_comp = instructions.Instructions(
        agent_name=entity_name,
        pre_act_label='\nRole playing instructions',
    )
```

**Add to components dictionary, BEFORE existing components:**
```python
# Assemble components
components_of_agent = {}

# Add Instructions first if enabled
if instructions_key:
    components_of_agent[instructions_key] = instructions_comp

# Then add existing components
components_of_agent[memory_key] = memory
components_of_agent[impe_memory_key] = impe_memory
components_of_agent[audience_eval_key] = audience_eval
components_of_agent[observation_to_memory_key] = observation_to_memory
```

**Add to params default factory:**
```python
'enable_instructions': True,
```

### Phase 2: Add SelfPerception Component

#### Step 2.1: Update Actor Prefab
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Add after imports:**
```python
from concordia.components.agent import question_of_recent_memories
```

**Add in `build()` method, after Instructions and before Cultural Norms (around line 104, after Instructions code):**
```python
# SelfPerception component (optional, but recommended)
enable_self_perception = bool(
    self.params.get('enable_self_perception', True)  # Default: enabled
)

self_perception_key = None
self_perception_comp = None
if enable_self_perception:
    self_perception_key = 'SelfPerception'
    self_perception_comp = question_of_recent_memories.SelfPerception(
        model=model,
        pre_act_label=f'\nQuestion: What kind of person is {entity_name}?\nAnswer',
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory (around line 36):**
```python
'enable_self_perception': True,
```

#### Step 2.2: Update Audience Prefab
**File:** `projects/impression_management_standard/simple_audience_prefab.py`

**Add after imports:**
```python
from concordia.components.agent import question_of_recent_memories
```

**Add in `build()` method, after Instructions:**
```python
# SelfPerception component (optional, but recommended)
enable_self_perception = bool(
    self.params.get('enable_self_perception', True)  # Default: enabled
)

self_perception_key = None
self_perception_comp = None
if enable_self_perception:
    self_perception_key = 'SelfPerception'
    self_perception_comp = question_of_recent_memories.SelfPerception(
        model=model,
        pre_act_label=f'\nQuestion: What kind of person is {entity_name}?\nAnswer',
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory:**
```python
'enable_self_perception': True,
```

### Phase 3: Add SituationPerception Component (Optional)

#### Step 3.1: Update Actor Prefab
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Add in `build()` method, after SelfPerception (after SelfPerception code):**
```python
# SituationPerception component (optional)
enable_situation_perception = bool(
    self.params.get('enable_situation_perception', False)  # Default: disabled
)

situation_perception_key = None
situation_perception_comp = None
if enable_situation_perception:
    situation_perception_key = 'SituationPerception'
    situation_perception_comp = question_of_recent_memories.SituationPerception(
        model=model,
        pre_act_label=f'\nQuestion: What kind of situation is {entity_name} in right now?\nAnswer',
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory:**
```python
'enable_situation_perception': False,
```

#### Step 3.2: Update Audience Prefab
**File:** `projects/impression_management_standard/simple_audience_prefab.py`

**Add in `build()` method, after SelfPerception (after SelfPerception code):**
```python
# SituationPerception component (optional)
enable_situation_perception = bool(
    self.params.get('enable_situation_perception', False)  # Default: disabled
)

situation_perception_key = None
situation_perception_comp = None
if enable_situation_perception:
    situation_perception_key = 'SituationPerception'
    situation_perception_comp = question_of_recent_memories.SituationPerception(
        model=model,
        pre_act_label=f'\nQuestion: What kind of situation is {entity_name} in right now?\nAnswer',
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory:**
```python
'enable_situation_perception': False,
```

### Phase 4: Add PersonBySituation Component (Optional)

#### Step 4.1: Update Actor Prefab
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Add in `build()` method, after SituationPerception:**
```python
# PersonBySituation component (optional, requires SelfPerception and SituationPerception)
enable_person_by_situation = bool(
    self.params.get('enable_person_by_situation', False)  # Default: disabled
)

person_by_situation_key = None
person_by_situation_comp = None
if enable_person_by_situation and self_perception_key and situation_perception_key:
    person_by_situation_key = 'PersonBySituation'
    person_by_situation_comp = question_of_recent_memories.PersonBySituation(
        model=model,
        components=[
            self_perception_key,
            situation_perception_key,
        ],
        pre_act_label=f'\nQuestion: What would a person like {entity_name} do in a situation like this?\nAnswer',
    )
elif enable_person_by_situation:
    # Warn if dependencies not met
    import warnings
    warnings.warn(
        f"PersonBySituation requires both SelfPerception and SituationPerception. "
        f"Disabling PersonBySituation for {entity_name}.",
        UserWarning
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory:**
```python
'enable_person_by_situation': False,
```

#### Step 4.2: Update Audience Prefab
**File:** `projects/impression_management_standard/simple_audience_prefab.py`

**Add in `build()` method, after SituationPerception (after SituationPerception code):**
```python
# PersonBySituation component (optional, requires SelfPerception and SituationPerception)
enable_person_by_situation = bool(
    self.params.get('enable_person_by_situation', False)  # Default: disabled
)

person_by_situation_key = None
person_by_situation_comp = None
if enable_person_by_situation and self_perception_key and situation_perception_key:
    person_by_situation_key = 'PersonBySituation'
    person_by_situation_comp = question_of_recent_memories.PersonBySituation(
        model=model,
        components=[
            self_perception_key,
            situation_perception_key,
        ],
        pre_act_label=f'\nQuestion: What would a person like {entity_name} do in a situation like this?\nAnswer',
    )
elif enable_person_by_situation:
    # Warn if dependencies not met
    import warnings
    warnings.warn(
        f"PersonBySituation requires both SelfPerception and SituationPerception. "
        f"Disabling PersonBySituation for {entity_name}.",
        UserWarning
    )
```

**Note:** Component addition to dictionary will be handled in Phase 6. For now, just add the component creation code above.

**Add to params default factory:**
```python
'enable_person_by_situation': False,
```

### Phase 5: Update Configuration

#### Step 5.1: Add CLI Arguments
**File:** `projects/impression_management_standard/config.py` (or wherever CLI args are parsed)

**Add to argument parser:**
```python
parser.add_argument(
    '--no_instructions',
    action='store_true',
    help='Disable Instructions component (role-playing context).'
)
parser.add_argument(
    '--no_self_perception',
    action='store_true',
    help='Disable SelfPerception component ("who am I?" questions).'
)
parser.add_argument(
    '--enable_situation_perception',
    action='store_true',
    help='Enable SituationPerception component ("what situation am I in?" questions).'
)
parser.add_argument(
    '--enable_person_by_situation',
    action='store_true',
    help='Enable PersonBySituation component ("what would I do?" reasoning). Requires --enable_situation_perception.'
)
```

#### Step 5.2: Update ConversationConfig
**File:** `projects/impression_management_standard/models.py`

**Add to `ConversationConfig` dataclass:**
```python
no_instructions: bool = False  # Whether to disable Instructions component
no_self_perception: bool = False  # Whether to disable SelfPerception component
enable_situation_perception: bool = False  # Whether to enable SituationPerception
enable_person_by_situation: bool = False  # Whether to enable PersonBySituation
```

#### Step 5.3: Update Simulation Config
**File:** `projects/impression_management_standard/simulation_config.py`

**Add to actor entity params (around line 88):**
```python
'enable_instructions': not config.no_instructions,
'enable_self_perception': not config.no_self_perception,
'enable_situation_perception': config.enable_situation_perception,
'enable_person_by_situation': config.enable_person_by_situation,
```

**Add to audience entity params (around line 108):**
```python
'enable_instructions': not config.no_instructions,
'enable_self_perception': not config.no_self_perception,
'enable_situation_perception': config.enable_situation_perception,
'enable_person_by_situation': config.enable_person_by_situation,
```

### Phase 6: Component Ordering and Dependencies

#### Step 6.1: Ensure Proper Component Order
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Replace the existing component assembly section (around line 187) with:**
```python
# Assemble components in order (dependencies first)
components_of_agent = {}

# 1. Instructions (first - provides experimental context)
if instructions_key:
    components_of_agent[instructions_key] = instructions_comp

# 2. Self-perception (can use traits, norms, memories)
if self_perception_key:
    components_of_agent[self_perception_key] = self_perception_comp

# 3. Situation perception (uses observations and memories)
if situation_perception_key:
    components_of_agent[situation_perception_key] = situation_perception_comp

# 4. Person-by-situation (depends on self and situation perception)
if person_by_situation_key:
    components_of_agent[person_by_situation_key] = person_by_situation_comp

# 5. Memory components (required for perception components)
components_of_agent[memory_key] = memory
components_of_agent[impe_memory_key] = impe_memory

# 6. Observation components (existing)
components_of_agent[observation_to_memory_key] = observation_to_memory

# 7. Other IMPE components (existing)
components_of_agent[actor_pf_key] = actor_pf
components_of_agent[reflection_key] = reflection

# 8. Cultural norms (existing, conditional)
if cultural_norms_key:
    components_of_agent[cultural_norms_key] = cultural_norms_comp

# 9. Personality traits (existing, conditional)
if personality_traits_key:
    components_of_agent[personality_traits_key] = personality_traits_comp
```

**Note:** This replaces the existing `components_of_agent = {...}` dictionary initialization. The order matters because:
- Instructions should be first (experimental context)
- SelfPerception and SituationPerception need memory components to exist (but memory is added after, which is fine - components access each other at runtime)
- PersonBySituation needs SelfPerception and SituationPerception to exist
- Cultural norms and traits can come after (they're used by perception components but don't depend on them)

**For Audience Prefab (`simple_audience_prefab.py`):**
Replace the component assembly section (around line 154) similarly, but include `audience_eval_key` instead of `actor_pf_key` and `reflection_key`:

```python
# Assemble components in order (dependencies first)
components_of_agent = {}

# 1. Instructions (first - provides experimental context)
if instructions_key:
    components_of_agent[instructions_key] = instructions_comp

# 2. Self-perception (can use traits, norms, memories)
if self_perception_key:
    components_of_agent[self_perception_key] = self_perception_comp

# 3. Situation perception (uses observations and memories)
if situation_perception_key:
    components_of_agent[situation_perception_key] = situation_perception_comp

# 4. Person-by-situation (depends on self and situation perception)
if person_by_situation_key:
    components_of_agent[person_by_situation_key] = person_by_situation_comp

# 5. Memory components (required for perception components)
components_of_agent[memory_key] = memory
components_of_agent[impe_memory_key] = impe_memory

# 6. Observation components (existing)
components_of_agent[observation_to_memory_key] = observation_to_memory

# 7. Audience evaluation component (existing)
components_of_agent[audience_eval_key] = audience_eval

# 8. Cultural norms (existing, conditional)
if cultural_norms_key:
    components_of_agent[cultural_norms_key] = cultural_norms_comp

# 9. Personality traits (existing, conditional)
if personality_traits_key:
    components_of_agent[personality_traits_key] = personality_traits_comp
```

### Phase 7: Testing and Validation

#### Step 7.1: Unit Tests
**File:** Create or update test file

**Test cases:**
- Instructions component is added correctly
- SelfPerception component is added when enabled
- SituationPerception component is added when enabled
- PersonBySituation component is added when enabled and dependencies met
- PersonBySituation is not added when dependencies missing
- Component ordering is correct
- Components work with existing IMPE components

#### Step 7.2: Integration Tests
- Run full conversation with all components enabled
- Run full conversation with only Instructions enabled
- Run full conversation with Instructions + SelfPerception
- Verify prompts include self-perception questions
- Verify prompts include situation perception (when enabled)
- Verify prompts include person-by-situation reasoning (when enabled)
- Check that components don't interfere with IMPE functionality

#### Step 7.3: Manual Verification
- Check generated prompts in logs
- Verify Instructions text appears in prompts
- Verify SelfPerception questions and answers appear
- Verify SituationPerception questions and answers appear (when enabled)
- Verify PersonBySituation reasoning appears (when enabled)
- Test with `--no_self_perception` flag
- Test with `--enable_situation_perception` flag
- Test with `--enable_person_by_situation` flag

## Error Prevention Checklist

### Before Implementation:
- [ ] Verify Instructions component doesn't conflict with existing components
- [ ] Check that SelfPerception can access memory (uses DEFAULT_MEMORY_COMPONENT_KEY)
- [ ] Verify both prefabs already have memory_key (AssociativeMemory) - they do
- [ ] Verify both prefabs already have observation_to_memory_key - they do
- [ ] Ensure SituationPerception can access observations (uses memory by default)
- [ ] Verify PersonBySituation dependencies are checked correctly
- [ ] Confirm component ordering doesn't break dependencies
- [ ] Test that components are added in correct order to dictionary

### During Implementation:
- [ ] Test each component addition independently
- [ ] Verify component keys don't conflict
- [ ] Check that optional components are truly optional
- [ ] Ensure PersonBySituation checks for dependencies
- [ ] Verify component order is correct

### After Implementation:
- [ ] Run full conversation test with all components enabled
- [ ] Run full conversation test with minimal components (Instructions only)
- [ ] Verify no duplicate questions in prompts
- [ ] Check that components integrate with IMPE components
- [ ] Verify CLI flags work correctly
- [ ] Test both actor and audience prefabs

## Potential Issues and Solutions

### Issue 1: Component Dependencies
**Problem:** PersonBySituation requires SelfPerception and SituationPerception.
**Solution:** Check for dependencies before adding PersonBySituation. Warn if dependencies missing.

### Issue 2: Component Ordering
**Problem:** Components must be in correct order for dependencies to work.
**Solution:** Build components dictionary in explicit order, ensuring dependencies come first.

### Issue 3: Memory Access
**Problem:** SelfPerception and SituationPerception need access to memory.
**Solution:** Both prefabs already have `memory_key` (AssociativeMemory) which is what `QuestionOfRecentMemories` uses by default. The components automatically access memory via `DEFAULT_MEMORY_COMPONENT_KEY`. No additional setup needed.

### Issue 4: Prompt Length
**Problem:** Adding multiple perception components increases prompt length.
**Solution:** Make components optional. Start with Instructions (required) and SelfPerception (recommended), add others as needed.

### Issue 5: Performance Impact
**Problem:** Each perception component makes an LLM call.
**Solution:**
- Make components optional
- Consider caching if needed
- Start with minimal set (Instructions + SelfPerception)

### Issue 6: Integration with IMPE Components
**Problem:** New components might interfere with IMPE functionality.
**Solution:**
- Test thoroughly
- Ensure IMPE components still work correctly
- Components are additive (provide context, don't replace functionality)

## Implementation Order

1. **Phase 1**: Add Instructions component (required, framework alignment)
2. **Phase 2**: Add SelfPerception component (recommended, improves consistency)
3. **Phase 3**: Add SituationPerception component (optional, test first)
4. **Phase 4**: Add PersonBySituation component (optional, most complex)
5. **Phase 5**: Add configuration options
6. **Phase 6**: Ensure proper component ordering
7. **Phase 7**: Testing and validation

## Success Criteria

- [ ] Instructions component appears in prompts (required)
- [ ] SelfPerception component appears when enabled (recommended)
- [ ] SituationPerception component appears when enabled (optional)
- [ ] PersonBySituation component appears when enabled and dependencies met (optional)
- [ ] Components don't interfere with existing IMPE functionality
- [ ] Component ordering is correct
- [ ] CLI flags work correctly
- [ ] Both actor and audience receive components appropriately
- [ ] Agents show improved self-awareness and consistency
- [ ] Framework alignment with standard Concordia patterns

## Notes

- **Instructions Component**: Should always be included (required for framework alignment)
- **SelfPerception**: Recommended for better character consistency
- **SituationPerception**: Optional, adds context understanding
- **PersonBySituation**: Optional, most complex, requires both SelfPerception and SituationPerception
- **Component Order**: Critical - dependencies must come before components that use them
- **Memory Access**: SelfPerception and SituationPerception automatically use memory via QuestionOfRecentMemories
- **Performance**: Each perception component adds an LLM call. Start minimal, add as needed.
- **Integration**: Components are additive - they provide context in `pre_act`, don't replace IMPE functionality

## Reference Implementation

See `concordia/prefabs/entity/basic_scripted.py` for a complete example of how these components are used together in a standard Concordia prefab.

## Summary of Key Points

### Component Dependencies
- **Instructions**: No dependencies (can be first)
- **SelfPerception**: Needs memory (already present in both prefabs)
- **SituationPerception**: Needs memory (already present in both prefabs)
- **PersonBySituation**: Requires both SelfPerception AND SituationPerception

### Component Ordering
The order in the `components_of_agent` dictionary matters:
1. Instructions (if enabled)
2. SelfPerception (if enabled)
3. SituationPerception (if enabled)
4. PersonBySituation (if enabled and dependencies met)
5. Memory components (required)
6. Observation components (existing)
7. IMPE components (existing)
8. Cultural norms (existing, conditional)
9. Personality traits (existing, conditional)

### Important Notes
- Both prefabs already have `memory_key` (AssociativeMemory) and `observation_to_memory_key` - no need to add these
- Components are added incrementally to the existing `components_of_agent` dictionary structure
- All new components are optional except Instructions (which should be enabled by default)
- PersonBySituation dependency check must happen before adding to dictionary
- Use `warnings.warn()` instead of `print()` for dependency warnings
