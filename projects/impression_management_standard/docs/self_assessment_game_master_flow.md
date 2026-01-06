# Self-Assessment Component: Game Master Action Flow

## Question

**If we implement self-assessment, does the game master use the revised action?**

## Answer

**YES** - The game master receives and uses the **revised action** (if revision occurred), not the original action.

## Flow Diagram

```
1. Engine calls: next_entity.act(action_spec)
   ↓
2. EntityAgent.act() calls: self._act_component.get_action_attempt()
   ↓
3. IMPESelfAssessmentComponent.get_action_attempt():
   - Calls base_act_component.get_action_attempt() → gets original
   - Assesses consistency
   - Revises if needed → gets revised
   - Returns FINAL response (original OR revised)
   ↓
4. EntityAgent.act() returns: action_attempt (the final response)
   ↓
5. Engine receives: raw_action = next_entity.act(...)
   ↓
6. Engine calls: self.resolve(game_master, putative_event=action)
   ↓
7. resolve() calls: game_master.observe(f'{PUTATIVE_EVENT_TAG} {action}')
   ↓
8. Game Master receives: The FINAL action (revised if revision occurred)
```

## Code Evidence

### Step 1: Entity.act() Returns What Component Returns

From `concordia/agents/entity_agent.py` (lines 171-184):
```python
def act(self, action_spec: entity.ActionSpec = entity.DEFAULT_ACTION_SPEC) -> str:
    # ...
    action_attempt = self._act_component.get_action_attempt(
        contexts, action_spec
    )
    # ...
    return action_attempt  # Returns whatever the act component returns
```

### Step 2: Self-Assessment Returns Final Response

From `self_assessment_component_plan.md` (line 86):
```python
6. **Return Final Response**:
   - Return either original or revised response in format: `DIALOGUE: ...\nBODY: ...`
```

The self-assessment component returns the **final** response (original if no revision, revised if revision occurred).

### Step 3: Engine Uses Returned Action

From `concordia/environment/engines/sequential.py` (lines 288-299):
```python
raw_action = next_entity.act(entity_spec_to_use)  # Gets final response
if next_entity.name in raw_action:
    action = raw_action
else:
    action = f'{next_entity.name}: {raw_action}'

self.resolve(game_master=game_master,
             putative_event=action,  # Uses the final action
             verbose=verbose)
```

### Step 4: Game Master Observes Final Action

From `concordia/environment/engines/sequential.py` (line 145):
```python
def resolve(self, game_master: entity_lib.Entity, putative_event: str, ...):
    game_master.observe(observation=f'{PUTATIVE_EVENT_TAG} {putative_event}')
    # Game master receives the final action (revised if revision occurred)
```

## Key Points

1. **Self-Assessment is Transparent**: The game master doesn't know if revision occurred
2. **Only Final Action is Visible**: Game master only sees the final response
3. **Original Action is Hidden**: If revision occurred, the original action is never sent to the game master
4. **Memory Handling**: The self-assessment component handles memory updates to ensure only the final action is stored

## Implications

### ✅ Benefits

- **Consistency**: Game master always receives consistent actions (aligned with traits/norms)
- **Transparency**: Game master doesn't need to know about self-assessment
- **Clean Interface**: Self-assessment is internal to the entity

### ⚠️ Considerations

- **Debugging**: If you want to see original vs. revised actions, you need to check assessment logs
- **Memory**: Only the final action is stored in entity memory (original is removed if revised)
- **Information Flow**: Original action is logged in information flow history (if enabled), but game master only sees final

## Example Scenario

**Original Action** (inconsistent):
```
DIALOGUE: "I maintain steady eye contact to show confidence"
BODY: "Maintains eye contact"
```

**Self-Assessment**:
- Consistency Score: 0.3 (contradicts "avoids eye contact" trait)
- Action: REVISE

**Revised Action** (consistent):
```
DIALOGUE: "I demonstrate competence through clear, direct communication"
BODY: "Looks away while speaking"
```

**Game Master Receives**:
```
[putative_event] John: DIALOGUE: I demonstrate competence through clear, direct communication
BODY: Looks away while speaking
```

The game master **never sees** the original action - only the revised one.

## Summary

**YES**, the game master uses the revised action (if revision occurred). The self-assessment component acts as a filter/wrapper that ensures only consistent actions reach the game master. The original action is discarded if revision occurs, and only the final (possibly revised) action is:

1. Returned from `entity.act()`
2. Passed to `engine.resolve()`
3. Observed by the game master via `PUTATIVE_EVENT_TAG`
4. Stored in entity memory

This design ensures that inconsistent actions never enter the conversation, maintaining agent behavior consistency throughout the simulation.
