# Feature Comparison: pe_conversation_openai (3).py vs impression_management_standard

This document compares the features in `pe_conversation_openai (3).py` with the current implementation in `impression_management_standard`, identifying what's implemented, what's different, and what needs to be added.

---

## 1. Error Handling and API Calls

### Feature in pe_conversation_openai (3).py:
- Custom `OpenAIError` exception class
- Specific exception types: `(OpenAIError, requests.exceptions.RequestException, TimeoutError)`
- `timeout=timeout_s` parameter in OpenAI API call

### Status in impression_management_standard:
**NOT IMPLEMENTED** - Uses generic exception handling

### Implementation:
The standard version uses Concordia's language model abstraction, so OpenAI-specific error handling is handled at a different layer. However, if direct OpenAI API calls are needed:

1. Add custom exception class to `setup.py` or a new `exceptions.py`:
```python
class OpenAIError(Exception):
    pass
```

2. Update error handling in language model setup (if using direct OpenAI calls):
```python
except (OpenAIError, requests.exceptions.RequestException, TimeoutError) as e:
    # handle
```

**Note**: Since `impression_management_standard` uses Concordia's language model interface, timeout handling may be managed at the framework level.

---

## 2. PersonalityTrait Dataclass with negative_assertion

### Feature in pe_conversation_openai (3).py:
```python
@dataclass
class PersonalityTrait:
    name: str
    assertion: str
    negative_assertion: Optional[str] = None  # NEW
```

### Status in impression_management_standard:
**NOT IMPLEMENTED** - The Concordia component only has:
```python
@dataclass
class PersonalityTrait:
    name: str
    assertion: str
```

### Implementation:
1. **Update Concordia component** (`concordia/components/agent/impression_management_pe.py`):
```python
@dataclass
class PersonalityTrait:
    """Personality trait definition."""
    name: str
    assertion: str
    negative_assertion: str | None = None  # ADD THIS
```

2. **Update constants** (`projects/impression_management_standard/constants.py`):
   - Add `negative_assertion` to each trait definition:
```python
PersonalityTrait(
    "Detail-focused",
    "I tend to focus on individual parts and details more than the big picture.",
    "I do not tend to focus on individual parts and details more than the big picture."  # ADD
),
```

3. **Update all trait definitions** in `constants.py` with negative assertions.

---

## 3. generate_parametric_traits() Function

### Feature in pe_conversation_openai (3).py:
```python
def generate_parametric_traits(trait_list: List[PersonalityTrait], is_audience: bool) -> Dict[str, int]:
    """Set traits to max (3) for audience, min (0) for actor with assertions rather than scores."""
    all_traits = []
    for t in trait_list:
        if is_audience:
            trait = t.assertion
        else:
            trait = t.negative_assertion
        all_traits.append(trait)
    return all_traits
```

### Status in impression_management_standard:
**NOT IMPLEMENTED** - Only has `generate_trait_scores()` which returns numeric scores (0-3)

### Implementation:
Add to `projects/impression_management_standard/utils.py`:
```python
def generate_parametric_traits(
    trait_list: list[PersonalityTrait],
    is_audience: bool,
) -> list[str]:
    """Generate trait assertions (audience: assertion, actor: negative_assertion).

    Returns a list of trait assertion strings rather than scores.
    """
    all_traits = []
    for t in trait_list:
        if is_audience:
            trait = t.assertion
        else:
            # Use negative_assertion if available, otherwise use assertion
            trait = t.negative_assertion if t.negative_assertion else t.assertion
        all_traits.append(trait)
    return all_traits
```

**Note**: This requires `negative_assertion` to be added to `PersonalityTrait` first (see #2).

---

## 4. Enhanced Prompt Header with World-Building Context

### Feature in pe_conversation_openai (3).py:
- Extensive world-building context (2A25, Cadens, Riffers)
- Interview context when enabled
- Structured prompt construction with norms and traits

### Status in impression_management_standard:
**PARTIALLY IMPLEMENTED** - Uses simpler prompt construction in components

### Differences:
- `pe_conversation_openai (3).py` has elaborate world-building narrative
- `impression_management_standard` uses more direct, component-based prompts

### Implementation:
If world-building context is desired, it can be added to:
- `CulturalNormsComponent.initialize_norms()` method
- `PersonalityTraitsComponent` initialization
- Or create a new `WorldContextComponent`

However, this may not be necessary if the current component-based approach is sufficient.

---

## 5. initialize_personality_traits() Method

### Feature in pe_conversation_openai (3).py:
```python
def initialize_personality_traits(self, traits: List[str]) -> None:
    """Set behaviour profile with personality traits for the agent."""
    # Sends initialization prompt to LLM
```

### Status in impression_management_standard:
**NOT IMPLEMENTED** - Traits are included in prompts but not separately initialized

### Implementation:
This could be added to `PersonalityTraitsComponent` in Concordia:
```python
def initialize_traits(self, model: language_model.LanguageModel, entity_name: str) -> None:
    """Send initialization prompt to set behavioral profile."""
    if not self.traits:
        return

    intro = "You are role-playing a person with the following stable behavioral profile."
    trait_list = "\n".join(f"- {s}" for s in self.traits)

    prompt = f"""{intro}

BEHAVIORAL PROFILE:
{trait_list}

INSTRUCTIONS:
- Treat this profile as stable across the entire interaction.
- Do not mention or explain the profile explicitly.
- Let it subtly shape wording, focus, interpretation, and responses.
- Behave naturally as a person with this profile would.
"""

    model.sample_text(prompt)
```

**Note**: This may be redundant if traits are already included in every prompt.

---

## 6. format_response() Method

### Feature in pe_conversation_openai (3).py:
```python
def format_response(self, raw_output: str) -> Tuple[str, str]:
    """Parse raw LLM output into dialogue and body language components."""
```

### Status in impression_management_standard:
**IMPLEMENTED** - As `parse_dialogue_and_body()` in `utils.py`:
```python
def parse_dialogue_and_body(response: str) -> tuple[str, str]:
    """Parse dialogue and body language from response."""
```

### Difference:
- Functionally equivalent, just different location (method vs utility function)

---

## 7. question_check() Method

### Feature in pe_conversation_openai (3).py:
```python
def question_check(self):
    personality_check = """Provide self-statements about who you are and your personality traits."""
    context_check = """What is this situation: summarize the topic of your conversation so far."""
    # Returns both responses
```

### Status in impression_management_standard:
**NOT IMPLEMENTED**

### Implementation:
This could be added as a diagnostic/debugging component:
```python
class PersonalityContextCheckComponent(entity_component.ObservingComponent):
    """Component that checks agent's self-awareness of personality and context."""

    def observe(self, context: entity_component.ComponentContextMapping) -> str:
        personality_prompt = "Provide self-statements about who you are and your personality traits."
        context_prompt = "What is this situation: summarize the topic of your conversation so far."

        personality_response = self._model.sample_text(personality_prompt)
        context_response = self._model.sample_text(context_prompt)

        return f"Personality: {personality_response}\nContext: {context_response}"
```

**Note**: This seems to be for debugging/validation purposes and may not be necessary for production.

---

## 8. audience_self_reflection() Method

### Feature in pe_conversation_openai (3).py:
```python
def audience_self_reflection(self, actor_utt: Utterance, audience_reply: Utterance, I_t: float) -> str:
    """Audience assesses and improves upon their own last response for appropriateness."""
    # Generates improved response aligned with norms and traits
```

### Status in impression_management_standard:
**PARTIALLY IMPLEMENTED** - `IMPESelfAssessmentComponent` provides similar functionality but:
- Uses consistency scoring (0-1) rather than direct improvement
- Wraps the act component rather than being called separately
- More structured assessment process

### Differences:
1. **pe_conversation_openai (3).py**: Direct self-reflection method that generates improved response
2. **impression_management_standard**: Self-assessment component that scores consistency and optionally revises

### Implementation:
The current `IMPESelfAssessmentComponent` is actually more sophisticated. However, if the direct reflection approach is preferred, it could be added to `IMPEAudienceActComponent`:
```python
def _self_reflect(
    self,
    original_response: str,
    actor_utterance: Utterance,
    I_t: float,
) -> str:
    """Generate improved response aligned with norms and traits."""
    # Similar to pe_conversation_openai (3).py implementation
```

**Recommendation**: Keep using `IMPESelfAssessmentComponent` as it's more robust.

---

## 9. actor_self_reflection() Method

### Feature in pe_conversation_openai (3).py:
```python
def actor_self_reflection(self, actor_utt: Utterance, aud_utt: Utterance) -> str:
    """Actor assesses and improves upon their own last response."""
```

### Status in impression_management_standard:
**PARTIALLY IMPLEMENTED** - `IMPESelfAssessmentComponent` can wrap `IMPEActComponent` for actors too

### Implementation:
Same as #8 - `IMPESelfAssessmentComponent` already provides this functionality when wrapping actor components.

---

## 10. Two-Stage Response Generation (Initial → Refined)

### Feature in pe_conversation_openai (3).py:
- `audience_evaluate_and_respond()` returns `(I_t, utt, final_utt)` - both initial and final
- `act_based_on_belief()` generates initial response, then calls `actor_self_reflection()` for refinement

### Status in impression_management_standard:
**IMPLEMENTED** - `IMPESelfAssessmentComponent` does this:
- Generates initial response via base component
- Assesses consistency
- Optionally generates revised response
- Returns final response

### Difference:
- `pe_conversation_openai (3).py`: Always generates refined response
- `impression_management_standard`: Only refines if consistency < threshold (configurable)

---

## 11. audience_evaluate_and_respond() Return Value

### Feature in pe_conversation_openai (3).py:
```python
return I_t, utt, final_utt  # Returns both initial and final utterances
```

### Status in impression_management_standard:
**NOT IMPLEMENTED** - `IMPEAudienceActComponent` only returns final response

### Implementation:
To track both initial and final responses, modify `IMPEAudienceActComponent` or add logging:
```python
# In IMPEAudienceActComponent or wrapper
initial_response = self._generate_initial_response(...)
final_response = self._self_reflect(initial_response, ...) if self_reflect_enabled else initial_response

# Store both in memory or return tuple
return I_t, initial_response, final_response
```

**Note**: This is mainly for logging/debugging. The current implementation may be sufficient.

---

## 12. actor_update_particles() Signature Change

### Feature in pe_conversation_openai (3).py:
```python
def actor_update_particles(self, turn: int, listener_utt: Utterance, pf_model: Optional[ParticleFilter] = None)
```
- Removed `goal_description` parameter
- Uses `_prompt_header()` instead

### Status in impression_management_standard:
**DIFFERENT** - Uses component-based approach where goal is accessed via memory component

### Implementation:
Not needed - the component-based architecture handles this differently (goal is in memory component, not passed as parameter).

---

## 13. act_based_on_belief() Signature Change

### Feature in pe_conversation_openai (3).py:
```python
def act_based_on_belief(self, turn: int, belief: float, audience_last_utt: Utterance) -> Utterance
```
- Added `audience_last_utt` parameter for self-reflection

### Status in impression_management_standard:
**DIFFERENT** - `IMPEActComponent` accesses conversation history via memory component, not parameters

### Implementation:
Not needed - the component architecture provides access to conversation history through the memory component.

---

## 14. TurnLog Dataclass Changes

### Feature in pe_conversation_openai (3).py:
- Removed `reflection_text`
- Added:
  - `audience_text0` and `audience_body0` (initial response)
  - `actor_personality_check` and `actor_context_check`
  - `audience_personality_check` and `audience_context_check`

### Status in impression_management_standard:
**PARTIALLY IMPLEMENTED** - `TurnLog` in `models.py` has:
- `reflection_text` (still present)
- Does NOT have initial response fields
- Does NOT have personality/context check fields

### Implementation:
Update `projects/impression_management_standard/models.py`:
```python
@dataclass
class TurnLog:
    time: str
    turn: int
    speaker: str
    listener: str
    speaker_text: str
    speaker_body: str
    audience_I: float
    audience_text: str  # Final response
    audience_body: str  # Final response
    audience_text0: str | None = None  # ADD: Initial response
    audience_body0: str | None = None  # ADD: Initial response
    actor_I_hat: float
    actor_pe: float
    reflection_text: str  # Keep or remove based on preference
    ess: float
    # Optional: Add personality/context checks if question_check is implemented
    actor_personality_check: str | None = None
    actor_context_check: str | None = None
    audience_personality_check: str | None = None
    audience_context_check: str | None = None
```

Then update `data_extraction.py` to populate these fields.

---

## 15. ConversationStudy.run() Changes

### Feature in pe_conversation_openai (3).py:
- Calls `initialize_personality_traits()` for both agents
- Calls `question_check()` for both agents
- Stores initial and final audience responses
- No `learning()` call (removed from flow)

### Status in impression_management_standard:
**DIFFERENT** - Uses Concordia simulation loop (`sim.play()`) which handles component lifecycle differently

### Implementation:
- `initialize_personality_traits()`: Can be added to component initialization (see #5)
- `question_check()`: Can be added as optional diagnostic component (see #7)
- Initial/final responses: Can be tracked via `IMPESelfAssessmentComponent` logging
- `learning()`: Currently not in the standard flow (may need to be added if reflections are desired)

---

## 16. Main Function - Parametric Trait Generation

### Feature in pe_conversation_openai (3).py:
- Uses `generate_parametric_traits()` (assertion-based)
- Passes trait assertions as lists, not scores

### Status in impression_management_standard:
**NOT IMPLEMENTED** - Uses `generate_trait_scores()` (score-based: 0-3)

### Implementation:
1. Add `generate_parametric_traits()` to `utils.py` (see #3)
2. Update `simulation_config.py` to use parametric traits when desired:
```python
# Option 1: Use parametric traits
if use_parametric_traits:
    actor_traits = utils.generate_parametric_traits(traits, is_audience=False)
    audience_traits = utils.generate_parametric_traits(traits, is_audience=True)
    # Pass as trait assertions list instead of scores
else:
    # Option 2: Use score-based traits (current)
    trait_scores_actor = utils.generate_trait_scores(rng, traits, is_audience=False)
    trait_scores_audience = utils.generate_trait_scores(rng, traits, is_audience=True)
```

3. Update prefab to accept either trait assertions or trait scores.

---

## 17. Particle Filter Resampling Method Call

### Feature in pe_conversation_openai (3).py:
```python
indices = pf_model.systematic_resample(weights)  # Note: no underscore
```

### Status in impression_management_standard:
**POTENTIAL BUG** - The method is `_systematic_resample()` (private) in `ParticleFilter` class

### Implementation:
This appears to be a bug in `pe_conversation_openai (3).py`. The correct call should be:
```python
indices = pf_model._systematic_resample(weights)
```

Or make the method public in `ParticleFilter`:
```python
def systematic_resample(self, weights: List[float]) -> List[int]:  # Remove underscore
```

---

## Summary

### Fully Implemented:
- ✅ Format response parsing (`parse_dialogue_and_body`)
- ✅ Self-reflection mechanism (`IMPESelfAssessmentComponent` - more sophisticated)
- ✅ Two-stage response generation (via self-assessment component)

### Partially Implemented / Different Approach:
- ⚠️ Prompt header system (component-based vs monolithic)
- ⚠️ Particle filter updates (component-based architecture)
- ⚠️ TurnLog structure (has reflection_text, missing initial response fields)

### Not Implemented:
- ❌ `negative_assertion` in `PersonalityTrait`
- ❌ `generate_parametric_traits()` function
- ❌ `initialize_personality_traits()` method
- ❌ `question_check()` method
- ❌ Initial/final response tracking in TurnLog
- ❌ Personality/context check fields in TurnLog

### Recommended Implementation Priority:
1. **High**: Add `negative_assertion` to `PersonalityTrait` (#2)
2. **High**: Add `generate_parametric_traits()` (#3)
3. **Medium**: Add initial/final response tracking (#11, #14)
4. **Low**: Add `question_check()` for debugging (#7)
5. **Low**: Add `initialize_personality_traits()` if needed (#5)

---

## Notes

- The `impression_management_standard` uses a component-based architecture (Concordia framework) which is more modular than the monolithic `pe_conversation_openai (3).py` approach.
- Some features in `pe_conversation_openai (3).py` may be redundant or less necessary in the component-based architecture.
- `IMPESelfAssessmentComponent` provides more sophisticated self-reflection than the simple methods in `pe_conversation_openai (3).py`.
- Consider whether world-building context (2A25, Cadens, Riffers) is necessary or if current prompts are sufficient.
