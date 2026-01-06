# Information Flow History Bank - Issues Found

**Date**: 2025-12-27
**File Reviewed**: `temp/2025-12-27_23-04-32/information_flow_history_20251227_230557.json`
**Total Interactions**: 71

## Critical Issues

### 1. Component Name Tracking Not Working

**Issue**: All 71 interactions have `component_name: null`

**Impact**:
- Cannot identify which component made each LLM call
- Cannot distinguish between calls from:
  - `IMPEActComponent` (actor action generation)
  - `IMPEAudienceEvaluationComponent` (evaluation generation)
  - `IMPEReflectionComponent` (reflection generation)
  - `IMPEActorParticleFilterComponent` (belief estimation)
  - Game master components
  - Other components

**Root Cause**:
- Thread-local context functions (`set_component_context`, `clear_component_context`) exist in `concordia/language_model/logging_wrapper.py`
- These functions are never called by the framework
- Components don't set context before making model calls
- `EntityAgentWithLogging` doesn't set context when calling component methods

**Example from Log**:
```json
{
  "agent_name": "Jane",
  "component_name": null,  // Should be "IMPEAudienceEvaluationComponent" or similar
  "method": "sample_text",
  "prompt": "...",
  "response": "0.8"
}
```

**Required Fix**:
- Integrate context setting into `EntityAgentWithLogging.act()`, `observe()`, `pre_act()`, `post_act()` methods
- OR have components set context themselves before calling the model
- OR use a context manager pattern to automatically set/clear context

---

### 2. Phase Tracking Not Working

**Issue**: All 71 interactions have `phase: null`

**Impact**:
- Cannot determine which simulation phase each call occurred in
- Cannot distinguish between:
  - `pre_act` (pre-action phase)
  - `act` (action phase)
  - `post_act` (post-action phase)
  - `observe` (observation phase)

**Root Cause**:
- Same as component name tracking - context is never set
- Phase information needs to be set when `EntityAgentWithLogging` calls component methods

**Example from Log**:
```json
{
  "agent_name": "John",
  "component_name": null,
  "phase": null,  // Should be "act", "pre_act", "post_act", or "observe"
  "method": "sample_text",
  "prompt": "...",
  "response": "..."
}
```

**Required Fix**:
- Set phase context in `EntityAgentWithLogging` methods:
  - `act()` → set phase to "act"
  - `observe()` → set phase to "observe"
  - `pre_act()` → set phase to "pre_act"
  - `post_act()` → set phase to "post_act"
- Clear context after method completes

---

## Minor Issues

### 3. Turn Tracking Timing

**Issue**: Turn numbers increment at the END of each step (in checkpoint callback)

**Current Behavior**:
- Turn 0: Initialization and first step interactions
- Turn 1: Second step interactions (after first step completes)
- Turn 2: Third step interactions (after second step completes)
- etc.

**Potential Confusion**:
- Interactions during a step use the turn number from BEFORE the increment
- For example, all interactions in "step 1" are labeled as turn 0
- This is technically correct but may be counterintuitive

**Example**:
- Step 1 starts → interactions logged with turn 0
- Step 1 ends → turn increments to 1
- Step 2 starts → interactions logged with turn 1
- Step 2 ends → turn increments to 2

**Impact**: Low - data is still usable, but turn numbers may be one less than expected

**Potential Fix**:
- Increment turns at the START of each step (before any interactions)
- OR document the current behavior clearly
- OR use "step" terminology instead of "turn" in the logs

---

## Working Correctly

### 4. Data Structure ✓

**Status**: Valid JSON, all required fields present

**Verified**:
- ✅ All interactions have `timestamp`
- ✅ All interactions have `agent_name`
- ✅ All interactions have `method` (`sample_text` or `sample_choice`)
- ✅ ✅ All interactions have `prompt` (full prompts captured)
- ✅ All interactions have `response` (full responses captured)
- ✅ All interactions have `kwargs` (model parameters)
- ✅ All interactions have `metadata`
- ✅ All interactions have `turn` (numbers increment correctly)
- ✅ JSON is well-formed and parseable

### 5. Agent Tracking ✓

**Status**: All agents correctly tracked

**Agents Found**:
- ✅ "Jane" (audience entity) - 13 interactions
- ✅ "IMPE Conversation Rules" (game master) - 48 interactions
- ✅ "John" (actor entity) - 10 interactions

**Total**: 71 interactions across 3 agents

### 6. Method Coverage ✓

**Status**: Both methods captured

**Methods Logged**:
- ✅ `sample_text` - 68 interactions
- ✅ `sample_choice` - 3 interactions

### 7. Metadata Capture ✓

**Status**: Metadata correctly captured

**For `sample_text`**:
- ✅ `max_tokens`, `terminators`, `temperature`, `top_p`, `top_k`, `timeout`, `seed`

**For `sample_choice`**:
- ✅ `seed`, `responses` (list of options)
- ✅ `metadata.index` (selected index)
- ✅ `metadata.info` (additional info)

---

## Summary

### Critical Issues (Must Fix)
1. **Component name tracking**: All `component_name` fields are null
2. **Phase tracking**: All `phase` fields are null

### Minor Issues (Consider Fixing)
3. **Turn tracking timing**: Turns increment at end of step (may be confusing)

### Working Correctly
- Data structure and JSON format
- Agent tracking
- Method coverage
- Metadata capture
- Prompt/response logging

---

## Recommended Next Steps

1. **Integrate Context Setting**:
   - Modify `EntityAgentWithLogging` to set component context and phase when calling component methods
   - Use context managers to ensure context is cleared after method completion

2. **Add Component Name Resolution**:
   - Components should identify themselves (e.g., via `__class__.__name__`)
   - OR use component keys to map to names

3. **Document Turn Tracking Behavior**:
   - Clarify that turns increment at end of step
   - OR change to increment at start of step

4. **Test Context Integration**:
   - Verify `component_name` and `phase` are populated after fixes
   - Test with a simple simulation to confirm

---

## Files to Modify

1. `concordia/agents/entity_agent_with_logging.py`
   - Add context setting in `act()`, `observe()`, `pre_act()`, `post_act()` methods

2. `concordia/language_model/logging_wrapper.py`
   - Already has context functions - no changes needed

3. `concordia/components/agent/impression_management_pe.py`
   - Optionally: components could set their own context before model calls

---

## Testing Checklist

After fixes are implemented:

- [ ] Run simulation with `--enable_info_flow_logging`
- [ ] Verify `component_name` is populated (not null)
- [ ] Verify `phase` is populated (not null)
- [ ] Check that component names match actual component classes
- [ ] Check that phases match simulation phases (pre_act, act, post_act, observe)
- [ ] Verify turn numbers are correct
- [ ] Verify all interactions are still captured
- [ ] Verify JSON structure is still valid
