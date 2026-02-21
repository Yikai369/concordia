# Implementation Plan: Enhanced Prompt Header with World-Building Context

## Overview

This plan implements the world-building context feature from `pe_conversation_openai (3).py` into `impression_management_standard`. The feature adds a fictional world setting (2A25, Cadens, Riffers) and interview context to agent prompts, creating a rich narrative framework for the simulation.

## Current State Analysis

### What Exists:
- `CulturalNormsComponent.get_norms_text()` - Returns norms with basic context ("alternative world in the year 3025")
- `IMPEActComponent._get_prompt_header()` - Combines norms and traits text
- `IMPEAudienceEvaluationComponent._get_prompt_header()` - Similar combination
- `IMPESelfAssessmentComponent._get_prompt_header()` - Similar combination
- Interview context is partially handled via `goal.role` in prompts

### What's Missing:
- **2A25 world-building narrative** (Cadens, Riffers, stigma dynamics)
- **Comprehensive world context** in prompt headers
- **Interview context integration** with world-building
- **Centralized world context component** or method

## Design Decisions

### Option 1: New WorldContextComponent (Recommended)
**Pros:**
- Modular and reusable
- Can be enabled/disabled independently
- Clean separation of concerns
- Easy to test

**Cons:**
- Requires new component creation
- More files to maintain

### Option 2: Extend CulturalNormsComponent
**Pros:**
- Fewer components
- World context naturally relates to norms

**Cons:**
- Mixes concerns (norms + world-building)
- Harder to disable world-building while keeping norms
- Less flexible

### Option 3: Add to Prompt Header Methods
**Pros:**
- Quick to implement
- No new components

**Cons:**
- Code duplication across multiple components
- Harder to maintain consistency
- Not reusable

**Decision: Use Option 1 (WorldContextComponent)** for modularity and maintainability.

## Implementation Plan

### Phase 1: Create WorldContextComponent

#### Step 1.1: Create Component Class
**File:** `concordia/components/agent/impression_management_pe.py`

**Add after `PersonalityTraitsComponent` class (around line 584):**

**Note:** Ensure `WorldContextComponent` is imported in prefab files that use it.

```python
class WorldContextComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for world-building context (2A25 setting)."""

  def __init__(
      self,
      enable_world_building: bool = True,
      enable_interview_context: bool = True,
      pre_act_label: str = 'World Context',
  ):
    """Initialize world context component.

    Args:
      enable_world_building: Whether to include 2A25 world-building narrative.
      enable_interview_context: Whether to include interview-specific context.
      pre_act_label: Label for pre_act output.
    """
    super().__init__(pre_act_label)
    self._enable_world_building = enable_world_building
    self._enable_interview_context = enable_interview_context

  def get_world_context_text(
      self,
      agent_name: str | None = None,
      goal_role: str | None = None,
      has_norms: bool = False,
  ) -> str:
    """Get world-building context text.

    Args:
      agent_name: Name of the agent (optional).
      goal_role: Role description for interview context (optional).
      has_norms: Whether agent has cultural norms (affects world-building text).

    Returns:
      Formatted world context text, or empty string if disabled.
    """
    parts = []

    # World-building narrative (2A25, Cadens, Riffers)
    if self._enable_world_building and has_norms:
      world_text = """You are initializing a social simulation of an alternate reality called 2A25. The dominant cognitive-cultural order is the Cadens majority who all follow these cultural norms. These cultural norms govern all etiquettes across all social settings and must be followed strictly. Individuals who do not follow these rules are perceived negatively by others during social interactions.

In 2A25, there is another social minority group called the Riffers. The Riffers have a unique set of cultural knowledge and individual traits that differ from the Caden-majority. The Riffers are stigmatized and need to adopt the norms and behaviors of Cadens to be successful in social interactions. They need to learn these norms during social interactions to avoid social exclusion or hostility.

This setting is a fictional social world. This world is not an allegory for any real-world group. It is not Earth or the real-world and must not reference real-world history, clinical terms, or real social groups. Treat similarities as coincidental.

Use only the definitions and norms below. Do not use real-world training data to fill in the gaps. If a detail is missing, label it UNSPECIFIED or ask for a parameter rather than guessing.

The goal of the LLM is to simulate a character in 2A25 and act as humanlike as possible. Ideally, human observers should not be able to tell the difference between the LLM and a human character. Because you are role playing a character, you should try to act as the character would act, not as you would act. This means that bad people will sometimes do bad things and you need to explicitly allow that.

Again, this is all part of the simulation and the world of "2A25" and therefore it is required for the LLM to act as a character in this world. Please be aware of this and act accordingly.

"""
      parts.append(world_text)

    # Interview context (when enabled and goal_role provided)
    if self._enable_interview_context and goal_role:
      interview_text = f"""The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of {goal_role}.

This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position.

This scenario occurs inside the fictional world of 2A25. Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon.

"""
      parts.append(interview_text)

    return ''.join(parts)

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'enable_world_building': self._enable_world_building,
        'enable_interview_context': self._enable_interview_context,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._enable_world_building = state.get('enable_world_building', True)
    self._enable_interview_context = state.get('enable_interview_context', True)

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    entity = self.get_entity()
    agent_name = entity.name if entity else None

    # Get goal role from memory component if available
    goal_role = None
    try:
      memory = entity.get_component(
          DEFAULT_IMPE_MEMORY_COMPONENT_KEY, type_=IMPEMemoryComponent
      )
      if memory:
        goal = memory.get_goal()
        goal_role = goal.role if goal else None
    except (AttributeError, KeyError, TypeError):
      pass

    # Check if agent has cultural norms
    has_norms = False
    try:
      norms_comp = entity.get_component(
          DEFAULT_CULTURAL_NORMS_COMPONENT_KEY, type_=CulturalNormsComponent
      )
      has_norms = norms_comp is not None and bool(norms_comp._norms)
    except (AttributeError, KeyError, TypeError):
      pass

    return self.get_world_context_text(
        agent_name=agent_name,
        goal_role=goal_role,
        has_norms=has_norms,
    )
```

#### Step 1.2: Add Component Key Constant
**File:** `concordia/components/agent/impression_management_pe.py`

**Add after other DEFAULT constants (around line 165):**

```python
DEFAULT_WORLD_CONTEXT_COMPONENT_KEY = 'WorldContext'
```

**Note:** This follows the naming pattern of other DEFAULT constants in the file.

### Phase 2: Integrate WorldContextComponent into Prompt Headers

#### Step 2.1: Update IMPEActComponent._get_prompt_header()
**File:** `concordia/components/agent/impression_management_pe.py`

**Modify `_get_prompt_header()` method (around line 1020):**

```python
def _get_prompt_header(self) -> str:
  """Get prompt header with world context, norms and traits."""
  header_parts = []
  entity = self.get_entity()
  agent_name = entity.name if entity else None

  # World context (if component exists)
  world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
  try:
    world_comp = entity.get_component(
        world_context_key, type_=WorldContextComponent
    )
    if world_comp:
      # Get goal role from memory component
      goal_role = None
      try:
        memory = entity.get_component(
            self._memory_component_key, type_=IMPEMemoryComponent
        )
        if memory:
          goal = memory.get_goal()
          goal_role = goal.role if goal else None
      except (AttributeError, KeyError, TypeError):
        pass

      # Check if agent has norms
      has_norms = False
      if self._cultural_norms_key:
        try:
          norms_comp = entity.get_component(
              self._cultural_norms_key, type_=CulturalNormsComponent
          )
          has_norms = norms_comp is not None and bool(norms_comp._norms)
        except (AttributeError, KeyError, TypeError):
          pass

      world_text = world_comp.get_world_context_text(
          agent_name=agent_name,
          goal_role=goal_role,
          has_norms=has_norms,
      )
      if world_text:
        header_parts.append(world_text)
  except (AttributeError, KeyError, TypeError):
    pass  # World context component not present, skip

  # Cultural norms
  if self._cultural_norms_key:
    norms_comp = entity.get_component(
        self._cultural_norms_key, type_=CulturalNormsComponent
    )
    if norms_comp:
      # Pass agent name to include full initialization context
      header_parts.append(norms_comp.get_norms_text(agent_name))

  # Personality traits
  if self._personality_traits_key:
    traits_comp = entity.get_component(
        self._personality_traits_key, type_=PersonalityTraitsComponent
    )
    if traits_comp:
      header_parts.append(traits_comp.get_traits_text())

  return '\n'.join(header_parts)
```

#### Step 2.2: Update IMPEAudienceEvaluationComponent._get_prompt_header()
**File:** `concordia/components/agent/impression_management_pe.py`

**Modify `_get_prompt_header()` method (around line 649):**

```python
def _get_prompt_header(self) -> str:
  """Get prompt header with world context, norms and traits."""
  header_parts = []
  entity = self.get_entity()
  agent_name = entity.name if entity else None

  # World context (if component exists)
  world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
  try:
    world_comp = entity.get_component(
        world_context_key, type_=WorldContextComponent
    )
    if world_comp:
      # Get goal role from memory component
      goal_role = None
      try:
        memory = entity.get_component(
            self._memory_component_key, type_=IMPEMemoryComponent
        )
        if memory:
          goal = memory.get_goal()
          goal_role = goal.role if goal else None
      except (AttributeError, KeyError, TypeError):
        pass

      # Check if agent has norms
      has_norms = False
      if self._cultural_norms_key:
        try:
          norms_comp = entity.get_component(
              self._cultural_norms_key, type_=CulturalNormsComponent
          )
          has_norms = norms_comp is not None and bool(norms_comp._norms)
        except (AttributeError, KeyError, TypeError):
          pass

      world_text = world_comp.get_world_context_text(
          agent_name=agent_name,
          goal_role=goal_role,
          has_norms=has_norms,
      )
      if world_text:
        header_parts.append(world_text)
  except (AttributeError, KeyError, TypeError):
    pass  # World context component not present, skip

  # Cultural norms
  if self._cultural_norms_key:
    norms_comp = entity.get_component(
        self._cultural_norms_key, type_=CulturalNormsComponent
    )
    if norms_comp:
      header_parts.append(norms_comp.get_norms_text(agent_name))

  # Personality traits
  if self._personality_traits_key:
    traits_comp = entity.get_component(
        self._personality_traits_key, type_=PersonalityTraitsComponent
    )
    if traits_comp:
      header_parts.append(traits_comp.get_traits_text())

  return '\n'.join(header_parts)
```

#### Step 2.3: Update IMPESelfAssessmentComponent._get_prompt_header()
**File:** `concordia/components/agent/impression_management_pe.py`

**Modify `_get_prompt_header()` method (around line 1193):**

```python
def _get_prompt_header(self) -> str:
  """Get prompt header with world context, norms and traits."""
  header_parts = []
  entity = self.get_entity()
  agent_name = entity.name if entity else None

  # World context (if component exists)
  world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
  try:
    world_comp = entity.get_component(
        world_context_key, type_=WorldContextComponent
    )
    if world_comp:
      # Get goal role from memory component
      goal_role = None
      try:
        memory = entity.get_component(
            self._memory_component_key, type_=IMPEMemoryComponent
        )
        if memory:
          goal = memory.get_goal()
          goal_role = goal.role if goal else None
      except (AttributeError, KeyError, TypeError):
        pass

      # Check if agent has norms
      has_norms = False
      if self._cultural_norms_key:
        try:
          norms_comp = entity.get_component(
              self._cultural_norms_key, type_=CulturalNormsComponent
          )
          has_norms = norms_comp is not None and bool(norms_comp._norms)
        except (AttributeError, KeyError, TypeError):
          pass

      world_text = world_comp.get_world_context_text(
          agent_name=agent_name,
          goal_role=goal_role,
          has_norms=has_norms,
      )
      if world_text:
        header_parts.append(world_text)
  except (AttributeError, KeyError, TypeError):
    pass  # World context component not present, skip

  # Cultural norms
  norms_text = ''
  if self._cultural_norms_key:
    norms_comp = entity.get_component(
        self._cultural_norms_key, type_=CulturalNormsComponent
    )
    if norms_comp:
      norms_text = norms_comp.get_norms_text(agent_name=agent_name) + '\n\n'

  # Personality traits
  traits_text = ''
  if self._personality_traits_key:
    traits_comp = entity.get_component(
        self._personality_traits_key, type_=PersonalityTraitsComponent
    )
    if traits_comp:
      traits_text = traits_comp.get_traits_text() + '\n\n'

  return ''.join(header_parts) + norms_text + traits_text
```

### Phase 3: Add WorldContextComponent to Prefabs

#### Step 3.1: Update impression_management_actor Prefab
**File:** `concordia/prefabs/entity/impression_management_actor.py`

**Add to `build()` method after personality traits component (around line 140):**

```python
# World Context component (optional)
world_context_key = None
enable_world_building = bool(
    self.params.get('enable_world_building', True)
)
enable_interview_context = bool(
    self.params.get('enable_interview_context', True)
)

if enable_world_building or enable_interview_context:
  world_context_key = impe_components.DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
  world_context_comp = impe_components.WorldContextComponent(
      enable_world_building=enable_world_building,
      enable_interview_context=enable_interview_context,
      pre_act_label='\nWorld Context',
  )
```

**Add to components dictionary (around line 195):**

```python
if world_context_key:
  components_of_agent[world_context_key] = world_context_comp
```

**Add to params default factory (around line 40):**

```python
'enable_world_building': True,
'enable_interview_context': True,
```

#### Step 3.2: Update simple_audience_prefab
**File:** `projects/impression_management_standard/simple_audience_prefab.py`

**Add to `build()` method after personality traits component (around line 101):**

```python
# World Context component (optional)
world_context_key = None
enable_world_building = bool(
    self.params.get('enable_world_building', True)
)
enable_interview_context = bool(
    self.params.get('enable_interview_context', True)
)

if enable_world_building or enable_interview_context:
  world_context_key = impe_components.DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
  world_context_comp = impe_components.WorldContextComponent(
      enable_world_building=enable_world_building,
      enable_interview_context=enable_interview_context,
      pre_act_label='\nWorld Context',
  )
```

**Add to components dictionary (around line 135):**

```python
if world_context_key:
  components_of_agent[world_context_key] = world_context_comp
```

**Add to params default factory (around line 39):**

```python
'enable_world_building': True,
'enable_interview_context': True,
```

### Phase 4: Update Configuration

#### Step 4.1: Add CLI Arguments
**File:** `projects/impression_management_standard/config.py`

**Add to `parse_arguments()` function:**

```python
parser.add_argument(
    '--no_world_building',
    action='store_true',
    help='Disable 2A25 world-building context (Cadens, Riffers narrative).'
)
parser.add_argument(
    '--no_interview_context',
    action='store_true',
    help='Disable interview-specific context in world-building.'
)
```

**Add to `ConversationConfig` dataclass in `models.py`:**

```python
no_world_building: bool = False  # Whether to disable world-building context
no_interview_context: bool = False  # Whether to disable interview context
```

**Update return statement in `parse_arguments()`:**

```python
return ConversationConfig(
    # ... existing fields ...
    no_world_building=args.no_world_building,
    no_interview_context=args.no_interview_context,
)
```

#### Step 4.2: Update Simulation Config
**File:** `projects/impression_management_standard/simulation_config.py`

**Add to actor entity params (around line 88):**

```python
'enable_world_building': not config.no_world_building,
'enable_interview_context': not config.no_interview_context,
```

**Add to audience entity params (around line 108):**

```python
'enable_world_building': not config.no_world_building,
'enable_interview_context': not config.no_interview_context,
```

### Phase 5: Testing and Validation

#### Step 5.1: Unit Tests
**File:** `concordia/components/agent/impression_management_pe_test.py` (or create new test file)

**Add tests for:**
- `WorldContextComponent.get_world_context_text()` with various combinations
- Integration with `IMPEActComponent._get_prompt_header()`
- Integration with `IMPEAudienceEvaluationComponent._get_prompt_header()`
- Integration with `IMPESelfAssessmentComponent._get_prompt_header()`
- Disabled world-building scenarios
- Interview context with and without world-building

#### Step 5.2: Integration Tests
- Run full conversation with world-building enabled
- Run full conversation with world-building disabled
- Verify prompts include world context when enabled
- Verify prompts exclude world context when disabled
- Check that interview context integrates correctly

#### Step 5.3: Manual Verification
- Check generated prompts in logs
- Verify world-building text appears in appropriate places
- Verify interview context appears when enabled
- Test with `--no_world_building` flag
- Test with `--no_interview_context` flag

## Error Prevention Checklist

### Before Implementation:
- [ ] Verify `WorldContextComponent` doesn't conflict with existing components
- [ ] Ensure component key constant doesn't conflict with existing keys
- [ ] Check that all prompt header methods are updated consistently
- [ ] Verify backward compatibility (works when component not present)

### During Implementation:
- [ ] Test each component update independently
- [ ] Verify exception handling for missing components
- [ ] Check that `has_norms` detection works correctly
- [ ] Ensure `goal_role` extraction works for both actor and audience

### After Implementation:
- [ ] Run full conversation test with world-building enabled
- [ ] Run full conversation test with world-building disabled
- [ ] Verify no duplicate world context in prompts
- [ ] Check that interview context only appears when appropriate
- [ ] Verify component state save/load works correctly

## Potential Issues and Solutions

### Issue 1: Component Order in Prompt
**Problem:** World context might appear after norms, but should appear before.
**Solution:** Ensure `WorldContextComponent` is added first in `header_parts` list.

### Issue 2: Duplicate Context
**Problem:** World context might be added multiple times if multiple components use it.
**Solution:** Each component checks for world context component independently, but only adds it once per prompt header call.

### Issue 3: Interview Context Without World-Building
**Problem:** Interview context references "Caden-majority" but world-building might be disabled.
**Solution:** Interview context should only reference Cadens if world-building is enabled. Update `get_world_context_text()` to conditionally include Caden references.

### Issue 4: Missing Goal Role
**Problem:** `goal.role` might be None or empty string.
**Solution:** Check for `goal_role` before including interview context.

### Issue 5: Performance Impact
**Problem:** World-building text is long and adds to every prompt.
**Solution:**
- Make it optional via flags
- Consider caching if needed
- Only include when norms are present (for world-building narrative)

## Refinements

### Refinement 1: Conditional Caden References
**Update `get_world_context_text()` interview section:**

```python
if self._enable_interview_context and goal_role:
  # Only reference Cadens if world-building is enabled
  caden_reference = ""
  if self._enable_world_building:
    caden_reference = " This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position."

  interview_text = f"""The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of {goal_role}.{caden_reference}

This scenario occurs inside the fictional world of 2A25. Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon.

"""
  parts.append(interview_text)
```

**Wait, this still references 2A25 even if world-building is disabled. Let me refine further:**

```python
if self._enable_interview_context and goal_role:
  interview_text = f"""The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of {goal_role}."""

  if self._enable_world_building:
    interview_text += """ This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position.

This scenario occurs inside the fictional world of 2A25. Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon."""
  else:
    interview_text += """ The interview is designed to evaluate whether the candidate is competent for the role."""

  interview_text += "\n\n"
  parts.append(interview_text)
```

### Refinement 2: Extract World-Building Text to Constants
**File:** `projects/impression_management_standard/constants.py` or create `world_context.py`

**Add constants for world-building text:**

```python
# World-building narrative text
WORLD_BUILDING_NARRATIVE = """You are initializing a social simulation of an alternate reality called 2A25. The dominant cognitive-cultural order is the Cadens majority who all follow these cultural norms. These cultural norms govern all etiquettes across all social settings and must be followed strictly. Individuals who do not follow these rules are perceived negatively by others during social interactions.

In 2A25, there is another social minority group called the Riffers. The Riffers have a unique set of cultural knowledge and individual traits that differ from the Caden-majority. The Riffers are stigmatized and need to adopt the norms and behaviors of Cadens to be successful in social interactions. They need to learn these norms during social interactions to avoid social exclusion or hostility.

This setting is a fictional social world. This world is not an allegory for any real-world group. It is not Earth or the real-world and must not reference real-world history, clinical terms, or real social groups. Treat similarities as coincidental.

Use only the definitions and norms below. Do not use real-world training data to fill in the gaps. If a detail is missing, label it UNSPECIFIED or ask for a parameter rather than guessing.

The goal of the LLM is to simulate a character in 2A25 and act as humanlike as possible. Ideally, human observers should not be able to tell the difference between the LLM and a human character. Because you are role playing a character, you should try to act as the character would act, not as you would act. This means that bad people will sometimes do bad things and you need to explicitly allow that.

Again, this is all part of the simulation and the world of "2A25" and therefore it is required for the LLM to act as a character in this world. Please be aware of this and act accordingly.

"""

# Interview context templates
INTERVIEW_CONTEXT_BASE = """The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of {role}."""

INTERVIEW_CONTEXT_WITH_WORLD = """ This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position.

This scenario occurs inside the fictional world of 2A25. Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon.

"""

INTERVIEW_CONTEXT_WITHOUT_WORLD = """ The interview is designed to evaluate whether the candidate is competent for the role.

"""
```

**Then update `WorldContextComponent.get_world_context_text()` to use these constants.**

### Refinement 3: Handle Empty Goal Role
**In `get_world_context_text()`:**

```python
# Interview context (when enabled and goal_role provided)
if self._enable_interview_context and goal_role and goal_role.strip():
  # ... rest of interview context code
```

## Implementation Order

1. **Phase 1**: Create `WorldContextComponent` class
2. **Phase 2**: Update prompt header methods (test each independently)
3. **Phase 3**: Add component to prefabs (test actor first, then audience)
4. **Phase 4**: Add configuration options
5. **Phase 5**: Testing and validation

## Success Criteria

- [ ] World-building context appears in prompts when enabled
- [ ] World-building context is absent when disabled
- [ ] Interview context integrates correctly with world-building
- [ ] Interview context works independently when world-building is disabled
- [ ] No duplicate context in prompts
- [ ] Backward compatible (works when component not present)
- [ ] CLI flags work correctly
- [ ] Both actor and audience receive world context appropriately

## Notes

- World-building context should only appear when cultural norms are present (for the Cadens narrative)
- Interview context can appear independently of world-building
- The component is optional and can be disabled via configuration
- All prompt header methods need consistent updates to avoid duplication
- Consider performance impact of long world-building text in every prompt

## Summary

This implementation plan provides a comprehensive approach to adding world-building context to the impression management simulation:

1. **Creates a new `WorldContextComponent`** that encapsulates world-building logic
2. **Integrates the component** into all prompt header methods (Act, AudienceEvaluation, SelfAssessment)
3. **Adds configuration options** via CLI flags and config dataclass
4. **Updates prefabs** to include the component with proper parameters
5. **Includes error handling** and backward compatibility
6. **Provides testing guidelines** and validation criteria

The plan is designed to be:
- **Modular**: Component-based design allows easy enable/disable
- **Backward compatible**: Works when component is not present
- **Consistent**: All prompt headers updated uniformly
- **Testable**: Clear testing strategy and success criteria
- **Maintainable**: Centralized world-building logic in one component

**Key Implementation Points:**
- World-building text only appears when norms are present
- Interview context can work independently
- Proper error handling for missing components
- Consistent exception handling (AttributeError, KeyError, TypeError)
- Component order: World Context → Cultural Norms → Personality Traits
