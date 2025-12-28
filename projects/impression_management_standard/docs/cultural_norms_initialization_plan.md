# Cultural Norms Initialization Plan

## Overview

This plan addresses the issue where the cultural norms initialization context ("You are in an alternative world in the year 3025...") is only sent once during initialization but not included in subsequent prompts. Since LLMs don't have persistent memory between API calls, this context needs to be included in every prompt.

## Problem Statement

**Current Behavior**:
- `CulturalNormsComponent.initialize_norms()` sends a one-time initialization prompt with full context
- `get_norms_text()` only returns the norms list without the initialization context
- The initialization context is lost after the first call

**Impact**:
- LLMs don't remember the initialization context between API calls
- Agents may not consistently follow the cultural norms framework
- The "alternative world in 3025" context is missing from prompts

## Solution

Modify `get_norms_text()` to include the full initialization text, ensuring the context is included in every prompt that uses norms.

## Implementation Steps

### Step 1: Update `get_norms_text()` Method

**Location**: `concordia/components/agent/impression_management_pe.py` (lines 262-270)

**Current Implementation**:
```python
def get_norms_text(self) -> str:
    """Format norms as prompt text."""
    if not self._norms:
      return ''
    lines = ['CULTURAL NORMS YOU FOLLOW:']
    for norm in self._norms:
      lines.append(f'- {norm.name}: {norm.description}')
    lines.append('')
    return '\n'.join(lines)
```

**New Implementation**:
```python
def get_norms_text(self, agent_name: str | None = None) -> str:
    """Format norms as prompt text with full initialization context.

    Args:
        agent_name: Name of the agent (optional, for initialization context).
                    If None, only returns norms list (backward compatible).

    Returns:
        Formatted text with initialization context and norms list.
    """
    if not self._norms:
      return ''

    # Build norms description
    norms_desc = '\n'.join(
        f'- {n.name}: {n.description}' for n in self._norms
    )

    # If agent_name is provided, include full initialization context
    if agent_name:
        return f"""You are {agent_name}. You are in an alternative world in the year 3025 where there is a new set of cultural norms. In all your interactions, you must follow these cultural norms:

{norms_desc}

If you fail to do so, you will be unsuccessful in your interactions and perceived negatively by others. Always follow these norms strictly.

"""

    # Backward compatible: return just norms list if no agent_name
    lines = ['CULTURAL NORMS YOU FOLLOW:']
    for norm in self._norms:
      lines.append(f'- {norm.name}: {norm.description}')
    lines.append('')
    return '\n'.join(lines)
```

### Step 2: Update Components That Use `get_norms_text()`

**Components to Update**:
1. `IMPEActComponent._get_prompt_header()` (line 764-779)
2. `IMPEAudienceEvaluationComponent._get_prompt_header()` (line 396-411)
3. `CulturalNormsComponent._make_pre_act_value()` (line 303-305)

**Changes Needed**:
- Pass `agent_name` (from `self.get_entity().name`) to `get_norms_text()`

**Example Update for `IMPEActComponent._get_prompt_header()`**:
```python
def _get_prompt_header(self) -> str:
    """Get prompt header with norms and traits."""
    header_parts = []
    if self._cultural_norms_key:
      norms_comp = self.get_entity().get_component(
          self._cultural_norms_key, type_=CulturalNormsComponent
      )
      if norms_comp:
        # Pass agent name to include full initialization context
        header_parts.append(norms_comp.get_norms_text(self.get_entity().name))
    if self._personality_traits_key:
      traits_comp = self.get_entity().get_component(
          self._personality_traits_key, type_=PersonalityTraitsComponent
      )
      if traits_comp:
        header_parts.append(traits_comp.get_traits_text())
    return '\n'.join(header_parts)
```

### Step 3: Update `_make_pre_act_value()` Method

**Location**: `concordia/components/agent/impression_management_pe.py` (line 303-305)

**Current Implementation**:
```python
def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    return self.get_norms_text()
```

**New Implementation**:
```python
def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    # Get agent name from entity if available
    entity = self.get_entity()
    agent_name = entity.name if entity else None
    return self.get_norms_text(agent_name)
```

### Step 4: Keep `initialize_norms()` for Backward Compatibility

**Note**: The `initialize_norms()` method can remain for backward compatibility, but it's no longer strictly necessary since the context is now included in every prompt. However, we can keep it as an optional initialization step.

## Testing Strategy

### Unit Tests

1. **Test `get_norms_text()` with agent_name**:
   - Verify it includes full initialization context
   - Verify norms list is included
   - Verify formatting is correct

2. **Test `get_norms_text()` without agent_name** (backward compatibility):
   - Verify it returns just norms list (old behavior)
   - Ensure no errors when agent_name is None

3. **Test prompt headers**:
   - Verify `IMPEActComponent._get_prompt_header()` includes full context
   - Verify `IMPEAudienceEvaluationComponent._get_prompt_header()` includes full context
   - Verify `CulturalNormsComponent._make_pre_act_value()` includes full context

### Integration Tests

1. **Test full conversation**:
   - Run simulation with cultural norms enabled
   - Verify prompts include initialization context in every turn
   - Verify agent behavior is consistent with norms

2. **Test without norms**:
   - Verify no errors when norms are None or empty
   - Verify prompts work correctly without norms

## Benefits

1. **Consistency**: Initialization context included in every prompt
2. **Reliability**: Agents consistently follow cultural norms framework
3. **Backward Compatible**: Optional agent_name parameter maintains compatibility
4. **No Breaking Changes**: Existing code continues to work

## Considerations

1. **Prompt Length**: Including full context increases prompt size, but this is necessary for consistency
2. **Agent Name**: Need to ensure entity name is available when calling `get_norms_text()`
3. **Backward Compatibility**: Make agent_name optional to avoid breaking existing code

## Implementation Order

1. Update `get_norms_text()` method signature and implementation
2. Update `_make_pre_act_value()` to pass agent name
3. Update `IMPEActComponent._get_prompt_header()` to pass agent name
4. Update `IMPEAudienceEvaluationComponent._get_prompt_header()` to pass agent name
5. Run tests to verify functionality
6. Test with full simulation

## Reference

- Current implementation: `concordia/components/agent/impression_management_pe.py`
  - `CulturalNormsComponent.get_norms_text()`: lines 262-270
  - `CulturalNormsComponent.initialize_norms()`: lines 272-287
  - `IMPEActComponent._get_prompt_header()`: lines 764-779
  - `IMPEAudienceEvaluationComponent._get_prompt_header()`: lines 396-411
- Original implementation: `projects/impression_management/pe_conversation_openai.py` line 402
