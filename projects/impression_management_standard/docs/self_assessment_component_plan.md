# Self-Assessment Component Plan

## Overview

The Self-Assessment Component ensures that agent responses align with their background information (personality traits, cultural norms, goals, and context). It acts as a quality control mechanism that:

1. **Assesses consistency** between generated responses and background information
2. **Provides feedback** on how to improve alignment
3. **Optionally revises** responses when inconsistencies are not tolerable

## Component Design

### Class: `IMPESelfAssessmentComponent`

**Type**: `ActingComponent` (wraps `IMPEActComponent`)

**Location**: `concordia/components/agent/impression_management_pe.py`

**Purpose**: Intercept and validate agent responses before they are returned as actions.

### Architecture Decision: Wrapper Pattern

Instead of modifying `IMPEActComponent` directly, we'll create a wrapper component that:
- Internally calls `IMPEActComponent` to generate the initial response
- Assesses the response for consistency
- Optionally generates a revised response
- Returns either the original or revised response

This maintains modularity and allows the self-assessment to be optional.

## Component Interface

### Initialization

```python
class IMPESelfAssessmentComponent(entity_component.ActingComponent):
    def __init__(
        self,
        model: language_model.LanguageModel,
        base_act_component: IMPEActComponent,  # The wrapped component
        memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        cultural_norms_key: str | None = None,
        personality_traits_key: str | None = None,
        context: bool = True,
        consistency_threshold: float = 0.7,  # Acceptable consistency score (0-1)
        enable_revision: bool = True,  # Whether to revise if inconsistent
    ):
```

### Parameters

- `base_act_component`: The `IMPEActComponent` instance to wrap
- `consistency_threshold`: Minimum consistency score (0-1) to accept without revision
- `enable_revision`: If `True`, generates revised response when consistency is below threshold

## Component Workflow

### `get_action_attempt()` Method

1. **Generate Initial Response**:
   - Call `base_act_component.get_action_attempt()` to get the original response
   - Parse the response to extract `DIALOGUE` and `BODY`

2. **Collect Context Information**:
   - Retrieve from memory: goal, conversation history, I_hat, reflections
   - Retrieve from components: cultural norms text, personality traits text
   - Build context summary of what the agent "knows" about itself

3. **Assess Consistency**:
   - Prompt LLM to evaluate consistency between response and background info
   - Get consistency score (0-1) and feedback comment
   - Determine if revision is needed (score < threshold)

4. **Revise if Necessary**:
   - If `enable_revision=True` and consistency < threshold:
     - Use feedback comment to generate revised response
     - Validate revised response (optional second check)
   - If `enable_revision=False` or consistency >= threshold:
     - Return original response

5. **Log Assessment Results**:
   - Log consistency score, feedback, and whether revision occurred
   - Store assessment record in memory (optional)

6. **Return Final Response**:
   - Return either original or revised response in format: `DIALOGUE: ...\nBODY: ...`

## Detailed Implementation

### Step 1: Generate Initial Response

```python
def get_action_attempt(
    self,
    context: entity_component.ComponentContextMapping,
    action_spec: entity_lib.ActionSpec,
) -> str:
    # Delegate to base component
    original_response = self._base_act_component.get_action_attempt(
        context, action_spec
    )

    # Parse response
    m1 = re.search(r'DIALOGUE:\s*(.*)', original_response)
    m2 = re.search(r'BODY:\s*(.*)', original_response)
    original_text = m1.group(1).strip() if m1 else original_response.strip()
    original_body = m2.group(1).strip() if m2 else ''
```

### Step 2: Collect Context Information

```python
# Get memory and goal
memory = self.get_entity().get_component(
    self._memory_component_key, type_=IMPEMemoryComponent
)
goal = memory.get_goal()
current_turn = len(memory.get_recent_conversation()) + 1

# Get norms and traits text
norms_text = ''
if self._cultural_norms_key:
    norms_comp = self.get_entity().get_component(
        self._cultural_norms_key, type_=CulturalNormsComponent
    )
    if norms_comp:
        norms_text = norms_comp.get_norms_text()

traits_text = ''
if self._personality_traits_key:
    traits_comp = self.get_entity().get_component(
        self._personality_traits_key, type_=PersonalityTraitsComponent
    )
    if traits_comp:
        traits_text = traits_comp.get_traits_text()

# Get recent context
pf_history = memory.get_pf_history()
refl_k = memory.get_recent_reflections(memory._recent_k)
conv_k = memory.get_recent_conversation()
I_hat = pf_history[-1].get('I_hat', 0.5) if pf_history else 0.5
```

### Step 3: Assess Consistency

```python
# Build assessment prompt
assessment_prompt = f"""{norms_text}{traits_text}
You are {self.get_entity().name}. Your goal: {goal.name}.
Goal definition: {goal.description}.

Recent context:
- Current belief (I_hat): {I_hat:.2f}
- Recent reflections: {chr(10).join(f"- {r.text}" for r in refl_k[-2:]) or "- (none)"}
- Recent conversation: {memory.format_conversation(conv_k[-2:])}

You generated this response:
DIALOGUE: {original_text}
BODY: {original_body}

Assess whether this response is consistent with:
1. Your personality traits (above)
2. Your cultural norms (above)
3. Your goal and current belief
4. Your recent reflections

Rate the consistency on a scale from 0.0 to 1.0, where:
- 1.0 = Fully consistent with all background information
- 0.5 = Partially consistent, some misalignment
- 0.0 = Completely inconsistent

Respond in this exact format:
CONSISTENCY_SCORE: <0.0-1.0>
IS_ACCEPTABLE: <yes/no>
FEEDBACK: <brief comment on what is inconsistent and how to fix it>
"""

assessment_raw = self._model.sample_text(assessment_prompt)

# Parse assessment
score_match = re.search(r'CONSISTENCY_SCORE:\s*([01](?:\.\d+)?)', assessment_raw)
acceptable_match = re.search(r'IS_ACCEPTABLE:\s*(yes|no)', assessment_raw, re.IGNORECASE)
feedback_match = re.search(r'FEEDBACK:\s*(.*?)(?:\n|$)', assessment_raw, re.DOTALL)

consistency_score = float(score_match.group(1)) if score_match else 0.5
is_acceptable = (acceptable_match.group(1).lower() == 'yes') if acceptable_match else False
feedback = feedback_match.group(1).strip() if feedback_match else 'No feedback provided'
```

### Step 4: Revise if Necessary

```python
final_text = original_text
final_body = original_body
was_revised = False

if not is_acceptable and self._enable_revision:
    # Generate revised response
    revision_prompt = f"""{norms_text}{traits_text}
You are {self.get_entity().name}. Your goal: {goal.name}.
Goal definition: {goal.description}.

Recent context:
- Current belief (I_hat): {I_hat:.2f}
- Recent reflections: {chr(10).join(f"- {r.text}" for r in refl_k[-2:]) or "- (none)"}

You previously generated this response:
DIALOGUE: {original_text}
BODY: {original_body}

However, this response was assessed as inconsistent with your background information.
Assessment feedback: {feedback}

Generate a REVISED response that:
1. Maintains the core message/intent of the original
2. Better aligns with your personality traits
3. Better follows your cultural norms
4. Better supports your goal achievement
5. Incorporates the feedback above

Output in this format exactly:
DIALOGUE: <revised one sentence>
BODY: <revised brief body language phrase>
"""

    revision_raw = self._model.sample_text(revision_prompt)
    m1 = re.search(r'DIALOGUE:\s*(.*)', revision_raw)
    m2 = re.search(r'BODY:\s*(.*)', revision_raw)
    final_text = m1.group(1).strip() if m1 else original_text
    final_body = m2.group(1).strip() if m2 else original_body
    was_revised = True

    # Update memory with revised utterance (remove original, add revised)
    # Note: The base component already added the original, so we need to handle this
    # Option: Don't let base component add to memory, do it here instead
```

### Step 5: Log Assessment

```python
# Log assessment results
self._logging_channel({
    'Key': 'Self-Assessment',
    'Consistency Score': consistency_score,
    'Is Acceptable': is_acceptable,
    'Was Revised': was_revised,
    'Feedback': feedback,
    'Original Response': f'DIALOGUE: {original_text}\nBODY: {original_body}',
    'Final Response': f'DIALOGUE: {final_text}\nBODY: {final_body}',
})

# Store assessment record in memory (optional)
# Could add: memory.add_assessment_record(turn, score, feedback, was_revised)
```

### Step 6: Return Final Response

```python
# Update memory with final utterance
# Remove the original utterance added by base component
# Add the final (possibly revised) utterance
memory = self.get_entity().get_component(
    self._memory_component_key, type_=IMPEMemoryComponent
)
if was_revised:
    # Remove last utterance (the original one)
    if memory._conversation:
        memory._conversation.pop()
    # Add revised utterance
    memory.add_utterance(current_turn, self.get_entity().name, final_text, final_body)

return f'DIALOGUE: {final_text}\nBODY: {final_body}'
```

## Integration with Actor Prefab

### Modified Actor Prefab

In `concordia/prefabs/entity/impression_management_actor.py`:

```python
# Create base IMPE Act component
base_act_component = impe_components.IMPEActComponent(
    model=model,
    memory_component_key=impe_memory_key,
    cultural_norms_key=cultural_norms_key,
    personality_traits_key=personality_traits_key,
    context=context,
)

# Wrap with self-assessment (optional, controlled by parameter)
if self.params.get('enable_self_assessment', False):
    act_component = impe_components.IMPESelfAssessmentComponent(
        model=model,
        base_act_component=base_act_component,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        consistency_threshold=self.params.get('consistency_threshold', 0.7),
        enable_revision=self.params.get('enable_revision', True),
    )
else:
    act_component = base_act_component
```

### New Prefab Parameters

Add to `Entity.params`:
- `enable_self_assessment`: bool = False (enable/disable self-assessment)
- `consistency_threshold`: float = 0.7 (minimum acceptable consistency)
- `enable_revision`: bool = True (whether to revise when inconsistent)

## Memory Extension (Optional)

### Add Assessment Records to Memory

In `IMPEMemoryComponent`, add:

```python
@dataclass
class AssessmentRecord:
    turn: int
    consistency_score: float
    is_acceptable: bool
    feedback: str
    was_revised: bool
    original_text: str
    final_text: str

# In IMPEMemoryComponent:
self._assessment_history: list[AssessmentRecord] = []

def add_assessment_record(
    self,
    turn: int,
    consistency_score: float,
    is_acceptable: bool,
    feedback: str,
    was_revised: bool,
    original_text: str,
    final_text: str,
) -> None:
    """Add assessment record."""
    self._assessment_history.append(
        AssessmentRecord(
            turn=turn,
            consistency_score=consistency_score,
            is_acceptable=is_acceptable,
            feedback=feedback,
            was_revised=was_revised,
            original_text=original_text,
            final_text=final_text,
        )
    )
```

## Component Lifecycle Integration

### Current Flow (without self-assessment):
1. `pre_observe()`: Extract observation
2. `post_observe()`: Update PF
3. `pre_act()`: Generate reflection
4. `act()`: Generate response → **RETURN**

### New Flow (with self-assessment):
1. `pre_observe()`: Extract observation
2. `post_observe()`: Update PF
3. `pre_act()`: Generate reflection
4. `act()`:
   - Generate initial response (via base component)
   - Assess consistency
   - Revise if needed
   - **RETURN** (original or revised)

**Key Point**: Self-assessment happens **within** the `act()` phase, so it doesn't change the component lifecycle ordering.

## Handling Memory Updates

### Challenge: Duplicate Utterances

The base `IMPEActComponent` adds the utterance to memory in `get_action_attempt()`. If we revise, we need to:
1. Remove the original utterance
2. Add the revised utterance

**Solution**: Modify `IMPEActComponent` to accept a flag that prevents memory update, OR handle memory update in the wrapper.

**Recommended**: Add optional parameter to `IMPEActComponent`:
```python
def get_action_attempt(
    self,
    context: entity_component.ComponentContextMapping,
    action_spec: entity_lib.ActionSpec,
    skip_memory_update: bool = False,  # New parameter
) -> str:
    # ... generate response ...
    if not skip_memory_update:
        memory.add_utterance(current_turn, self.get_entity().name, text, body)
    return f'DIALOGUE: {text}\nBODY: {body}'
```

Then in wrapper:
```python
# Call base component with skip_memory_update=True
original_response = self._base_act_component.get_action_attempt(
    context, action_spec, skip_memory_update=True
)
# ... assess and revise ...
# Add final utterance to memory
memory.add_utterance(current_turn, self.get_entity().name, final_text, final_body)
```

## Testing Strategy

### Unit Tests

1. **Consistency Assessment**:
   - Test with consistent response → should return high score
   - Test with inconsistent response → should return low score
   - Test parsing of assessment output

2. **Revision Logic**:
   - Test that revision occurs when score < threshold
   - Test that original is kept when score >= threshold
   - Test that revision is skipped when `enable_revision=False`

3. **Memory Handling**:
   - Test that only final utterance is in memory
   - Test that original is removed when revised

### Integration Tests

1. **Full Turn with Self-Assessment**:
   - Generate response
   - Assess consistency
   - Revise if needed
   - Verify final response is used in conversation

2. **Consistency Over Multiple Turns**:
   - Track consistency scores over time
   - Verify improvements in consistency

## Configuration Options

### Command-Line Arguments

Add to `ConversationConfig`:
- `--enable_self_assessment`: Enable self-assessment component
- `--consistency_threshold`: Minimum acceptable consistency (default: 0.7)
- `--disable_revision`: Disable revision (only assess, don't revise)

### Default Behavior

- **Default**: Self-assessment **disabled** (backward compatible)
- When enabled: Threshold = 0.7, revision enabled

## Benefits

1. **Consistency Enforcement**: Ensures agent behavior aligns with stated traits/norms
2. **Quality Control**: Catches inconsistencies before they enter conversation
3. **Adaptive Behavior**: Agent learns to generate more consistent responses
4. **Debugging**: Assessment records help identify when/why inconsistencies occur
5. **Modularity**: Can be enabled/disabled without changing other components

## Potential Issues & Solutions

### Issue 1: Response Delay
- **Problem**: Self-assessment adds LLM calls, increasing latency
- **Solution**: Make it optional, use faster model for assessment, or cache assessments

### Issue 2: Over-Correction
- **Problem**: Revision might change meaning too much
- **Solution**: Include constraint in revision prompt to "maintain core message"

### Issue 3: Assessment Quality
- **Problem**: LLM might not assess accurately
- **Solution**: Use structured output format, add examples, or use separate assessment model

### Issue 4: Memory Race Conditions
- **Problem**: Base component and wrapper both updating memory
- **Solution**: Use `skip_memory_update` flag or handle memory in wrapper only

## Implementation Phases

### Phase 1: Basic Assessment
- Implement `IMPESelfAssessmentComponent` with assessment only (no revision)
- Add logging
- Unit tests

### Phase 2: Revision
- Add revision logic
- Handle memory updates correctly
- Integration tests

### Phase 3: Memory Extension
- Add `AssessmentRecord` to memory
- Store assessment history
- Add retrieval methods

### Phase 4: Integration
- Update actor prefab
- Add command-line arguments
- Update documentation

## Example Usage

```python
# In simulation_config.py
actor_params = {
    'name': 'John',
    'enable_self_assessment': True,
    'consistency_threshold': 0.75,
    'enable_revision': True,
    # ... other params ...
}
```

## Summary

The Self-Assessment Component provides a quality control layer that ensures agent responses align with their background information. It uses a wrapper pattern to maintain modularity, assesses consistency via LLM evaluation, and optionally revises responses when inconsistencies are detected. The component integrates seamlessly into the existing component lifecycle without requiring changes to other components.
