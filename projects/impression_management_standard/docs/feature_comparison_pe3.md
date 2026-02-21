# Feature Comparison: pe_conversation_openai (3).py vs impression_management_standard

This document compares the features in `pe_conversation_openai (3).py` with the current implementation in `impression_management_standard`, identifying what's implemented, what's different, and what needs to be added.

---

## 1. Error Handling and API Calls

### Explanation:
This feature provides more granular error handling for OpenAI API calls. Instead of catching all exceptions generically, it distinguishes between different types of errors:
- **OpenAIError**: Custom exception for OpenAI-specific errors
- **RequestException**: Network/HTTP errors from the requests library
- **TimeoutError**: Timeout errors when API calls exceed the specified duration

The `timeout` parameter ensures that API calls don't hang indefinitely, which is important for long-running simulations. This allows the system to:
- Retry on transient network errors
- Fail fast on timeouts
- Provide more specific error messages for debugging
- Handle rate limiting and API errors differently

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

### Explanation:
This feature extends the `PersonalityTrait` dataclass to include a `negative_assertion` field, which stores the opposite statement of the trait. For example:
- **Assertion**: "I tend to focus on individual parts and details more than the big picture."
- **Negative assertion**: "I do not tend to focus on individual parts and details more than the big picture."

This is useful for **parametric trait generation** (see #3), where:
- **Audience agents** (interviewers) use the positive assertion (they have the trait)
- **Actor agents** (interviewees) use the negative assertion (they don't have the trait)

This allows the system to generate trait-based prompts that explicitly state what the agent is NOT, which can be more effective than just omitting the trait. It enables a clearer distinction between audience and actor personality profiles without needing separate trait definitions.

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

### Explanation:
This function generates personality trait **assertions** (text statements) rather than numeric **scores** (0-3). It returns a list of trait assertion strings that can be directly included in prompts.

**How it works:**
- For **audience agents**: Returns the positive `assertion` (e.g., "I tend to focus on individual parts...")
- For **actor agents**: Returns the `negative_assertion` (e.g., "I do not tend to focus on individual parts...")

**Why it's useful:**
- **More explicit**: Directly states what the agent is/isn't, rather than relying on numeric scores
- **Better for prompts**: LLMs respond better to explicit statements than numeric scores
- **Clearer distinction**: Makes the difference between audience and actor personalities more obvious
- **Parametric control**: Allows fine-grained control over which traits are expressed as positive vs negative

This is an alternative to the score-based approach (`generate_trait_scores()`), which assigns numeric values (0-3) that must be interpreted in prompts.

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

### Explanation:
This feature adds extensive world-building narrative context to agent prompts, creating a fictional setting called "2A25" with:
- **Cadens**: The dominant cultural majority who follow specific cultural norms
- **Riffers**: A stigmatized minority group that needs to learn Caden norms to succeed
- **Interview context**: Corporate interview setting within this fictional world

**Purpose:**
- **Immersion**: Creates a rich narrative context that helps LLMs maintain consistent character behavior
- **Cultural framing**: Frames cultural norms as part of a fictional world, avoiding real-world associations
- **Stigma simulation**: Allows simulation of impression management scenarios where one group must adapt to another's norms
- **Consistency**: Helps agents maintain consistent behavior by grounding them in a specific world context

**How it works:**
The prompt header is constructed with:
1. World-building narrative (2A25 setting)
2. Cultural norms (as Caden majority rules)
3. Interview context (corporate setting)
4. Personality traits
5. Goal information

This creates a comprehensive context that shapes all agent responses throughout the conversation.

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

### Explanation:
This method sends a **one-time initialization prompt** to the LLM before the conversation begins, establishing the agent's behavioral profile. It tells the LLM to:
- Treat the personality traits as a stable behavioral profile
- Not mention or explain the profile explicitly
- Let it subtly shape wording, focus, interpretation, and responses
- Behave naturally as a person with this profile would

**Purpose:**
- **Priming**: Sets up the LLM's "personality" before any conversation turns
- **Consistency**: Helps maintain consistent personality across the entire interaction
- **Subtle influence**: Encourages the traits to influence behavior naturally rather than explicitly

**When it's used:**
Called once at the start of the conversation, before any turns. This is separate from including traits in every prompt - it's a "setup" phase that establishes the agent's character.

**Potential benefit:**
Some LLMs may respond better to a dedicated initialization phase rather than just including traits in every prompt. However, this may be redundant if traits are already included in every prompt.

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

### Explanation:
This utility function parses the LLM's raw text output into two components:
1. **Dialogue**: The spoken text (what the agent says)
2. **Body language**: Non-verbal description (posture, eye contact, gestures, etc.)

**How it works:**
The LLM is instructed to format responses as:
```
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
```

The function uses regex to extract these components:
- Searches for `DIALOGUE:` followed by text
- Searches for `BODY:` followed by text
- Falls back to treating the entire response as dialogue if format isn't found

**Why it's useful:**
- **Structured output**: Ensures consistent parsing of multi-part responses
- **Body language tracking**: Captures non-verbal communication which is important for impression management
- **Robust parsing**: Handles cases where LLM doesn't follow format exactly

**When it's used:**
Called after every LLM response to extract the dialogue and body language components before storing them in memory or returning them as an utterance.

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

### Explanation:
This method performs **diagnostic checks** to verify that the agent has internalized its personality traits and understands the conversation context. It asks the agent two questions:
1. **Personality check**: "Provide self-statements about who you are and your personality traits."
2. **Context check**: "What is this situation: summarize the topic of your conversation so far."

**Purpose:**
- **Validation**: Verifies that the agent actually "knows" its personality traits (not just following prompts)
- **Context awareness**: Checks if the agent understands what's happening in the conversation
- **Debugging**: Helps identify when agents lose track of their identity or context
- **Quality control**: Can detect when agents are not properly maintaining their character

**How it works:**
Called at the start of each turn (before the agent acts), sends two prompts to the LLM and returns both responses. These responses can be logged to track agent self-awareness over time.

**When it's useful:**
- **Research**: Understanding how well agents maintain personality consistency
- **Debugging**: Identifying when agents drift from their intended behavior
- **Validation**: Ensuring agents are properly initialized with their traits

**Note**: This is primarily a diagnostic/debugging tool and may not be necessary for production use, but can be valuable for research and development.

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

### Explanation:
This method allows the **audience agent** (interviewer) to review and improve its own response after generating it. It's a form of **self-critique** that ensures the response aligns with:
- Cultural norms the audience should follow
- Personality traits of the audience
- The internal evaluation score (I_t) that was assigned
- Contextual appropriateness

**How it works:**
1. After generating an initial response, the audience agent reviews it
2. The method prompts the LLM to generate an improved version
3. The improved response better aligns with norms, traits, and the evaluation score
4. The improved response replaces the original

**Purpose:**
- **Quality control**: Ensures audience responses are consistent with their background
- **Norm compliance**: Helps audience agents follow cultural norms more strictly
- **Trait consistency**: Maintains personality trait expression in responses
- **Score alignment**: Ensures the response tone matches the evaluation score (e.g., low score = critical response)

**When it's used:**
Called immediately after `audience_evaluate_and_respond()` generates the initial response, before the response is returned or stored.

**Difference from IMPESelfAssessmentComponent:**
- This method: Directly generates an improved response
- IMPESelfAssessmentComponent: Scores consistency (0-1) and only revises if below threshold

### Feature in pe_conversation_openai (3).py:
```python
def audience_self_reflection(self, actor_utt: Utterance, audience_reply: Utterance, I_t: float) -> str:
    """Audience assesses and improves upon their own last response for appropriateness."""
    # Generates improved response aligned with norms and traits
```

### Status in impression_management_standard:
**IMPLEMENTED** - `IMPESelfAssessmentComponent` provides similar functionality:
- Uses consistency scoring (0-1) rather than direct improvement
- Wraps the act component rather than being called separately
- More structured assessment process
- **Now available for both actor and audience** (as of latest update)

### Differences:
1. **pe_conversation_openai (3).py**: Direct self-reflection method that generates improved response
2. **impression_management_standard**: Self-assessment component that scores consistency and optionally revises

### Implementation:
The current `IMPESelfAssessmentComponent` is actually more sophisticated. It has been updated to:
- Accept any `ActingComponent` (not just `IMPEActComponent`)
- Work with `SimpleAudienceActComponent` for audience agents
- Handle cases where agents don't have particle filters (uses evaluation score for audience)
- Retrieve revised responses from conversation history

**Status**: ✅ **FULLY IMPLEMENTED FOR BOTH ACTOR AND AUDIENCE**

---

## 9. actor_self_reflection() Method

### Explanation:
This method allows the **actor agent** (interviewee) to review and improve its own response after generating it. Similar to `audience_self_reflection()`, but focused on the actor's personality traits rather than cultural norms.

**How it works:**
1. After generating an initial response, the actor agent reviews it
2. The method prompts the LLM to generate an improved version
3. The improved response better aligns with the actor's personality traits
4. The improved response replaces the original

**Purpose:**
- **Trait consistency**: Ensures actor responses reflect their personality traits
- **Character maintenance**: Helps actors maintain consistent character behavior
- **Quality control**: Improves response quality by self-critique
- **Natural expression**: Encourages traits to influence responses naturally

**When it's used:**
Called immediately after `act_based_on_belief()` generates the initial response, before the response is returned or stored.

**Difference from audience reflection:**
- Actor reflection focuses on personality traits
- Audience reflection focuses on cultural norms AND traits

### Feature in pe_conversation_openai (3).py:
```python
def actor_self_reflection(self, actor_utt: Utterance, aud_utt: Utterance) -> str:
    """Actor assesses and improves upon their own last response."""
```

### Status in impression_management_standard:
**IMPLEMENTED** - `IMPESelfAssessmentComponent` can wrap `IMPEActComponent` for actors

### Implementation:
Same as #8 - `IMPESelfAssessmentComponent` provides this functionality for both actor and audience components. It has been updated to work with any `ActingComponent`, including:
- `IMPEActComponent` (for actors)
- `SimpleAudienceActComponent` (for audience)

**Status**: ✅ **FULLY IMPLEMENTED FOR BOTH ACTOR AND AUDIENCE**

---

## 10. Two-Stage Response Generation (Initial → Refined)

### Explanation:
This feature implements a **two-stage response generation process** where agents:
1. **Stage 1**: Generate an initial response based on the prompt
2. **Stage 2**: Review and refine the response to better align with background information (traits, norms, goals)

**How it works:**
- **For audience**: Generates initial evaluation response, then self-reflects to improve alignment with norms/traits
- **For actor**: Generates initial action response, then self-reflects to improve alignment with traits

**Purpose:**
- **Quality improvement**: Second pass allows agents to catch inconsistencies and improve responses
- **Consistency enforcement**: Ensures responses align with agent's background information
- **Better alignment**: Helps agents maintain character consistency throughout the conversation
- **Error correction**: Allows agents to fix mistakes or misalignments in their initial response

**Benefits:**
- More consistent agent behavior
- Better adherence to personality traits and cultural norms
- Higher quality responses
- More reliable character maintenance

**Trade-offs:**
- **Cost**: Requires two LLM calls per response (more expensive)
- **Time**: Takes longer to generate responses
- **Complexity**: More complex code and potential failure points

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

### Explanation:
This feature modifies the return value of `audience_evaluate_and_respond()` to include **both the initial and final (refined) responses**, rather than just the final response.

**What it returns:**
- `I_t`: The true evaluation score (0-1)
- `utt`: The initial response (before self-reflection)
- `final_utt`: The final response (after self-reflection/refinement)

**Purpose:**
- **Logging**: Allows tracking of how responses change during refinement
- **Analysis**: Enables comparison of initial vs final responses to measure improvement
- **Debugging**: Helps identify when and how responses are being refined
- **Research**: Useful for studying the effect of self-reflection on response quality

**Use cases:**
- Measuring the impact of self-reflection on response quality
- Identifying cases where refinement significantly changes the response
- Understanding when refinement is most beneficial
- Debugging issues with self-reflection process

**When it's useful:**
- Research on self-reflection effectiveness
- Debugging refinement issues
- Analyzing response quality improvements
- Understanding agent behavior changes

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

### Explanation:
This change modifies the `actor_update_particles()` method signature to remove the `goal_description` parameter and instead use the agent's `_prompt_header()` method to access goal information.

**What changed:**
- **Before**: `goal_description` was passed as a parameter
- **After**: Goal information is accessed via `_prompt_header()` which includes norms, traits, and goal

**Why the change:**
- **Consistency**: All prompt construction goes through `_prompt_header()`
- **Redundancy reduction**: Avoids passing the same information multiple ways
- **Centralization**: Single source of truth for prompt construction
- **Flexibility**: `_prompt_header()` can include additional context (norms, traits) automatically

**How it works:**
Instead of passing `goal_description` separately, the method calls `self._prompt_header()` which constructs a comprehensive prompt header including:
- Cultural norms (if applicable)
- Personality traits (if applicable)
- Goal name and description
- Context information

This ensures the particle filter update uses the same prompt structure as other agent methods.

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

### Explanation:
This change adds the `audience_last_utt` parameter to `act_based_on_belief()` to provide the most recent audience utterance for self-reflection purposes.

**What changed:**
- **Before**: Method only had `turn` and `belief` parameters
- **After**: Added `audience_last_utt` parameter containing the audience's last response

**Why it's needed:**
- **Self-reflection**: The actor needs the audience's last response to perform self-reflection
- **Context**: The actor's self-reflection should consider what the audience just said
- **Alignment**: Helps ensure the actor's refined response is appropriate given the audience's feedback

**How it's used:**
1. Actor generates initial response based on belief
2. Actor calls `actor_self_reflection()` with:
   - The actor's initial response
   - The audience's last utterance (for context)
3. Self-reflection generates improved response considering both

**Purpose:**
- **Contextual refinement**: Self-reflection considers the conversation context
- **Appropriate responses**: Ensures refined responses are relevant to what was just said
- **Better alignment**: Helps actor respond appropriately to audience feedback

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

### Explanation:
This feature modifies the `TurnLog` dataclass to track additional information about each conversation turn:
1. **Initial vs Final Responses**: Tracks both the initial response and the final (refined) response
2. **Personality/Context Checks**: Stores the results of `question_check()` calls
3. **Removed Reflection Text**: Removes the reflection field (possibly because reflections are handled differently)

**New fields:**
- `audience_text0` / `audience_body0`: Initial audience response (before refinement)
- `actor_personality_check`: Result of actor's personality self-check
- `actor_context_check`: Result of actor's context understanding check
- `audience_personality_check`: Result of audience's personality self-check
- `audience_context_check`: Result of audience's context understanding check

**Purpose:**
- **Response tracking**: Compare initial vs final responses to measure refinement impact
- **Self-awareness monitoring**: Track how well agents maintain awareness of their personality and context
- **Research data**: Collect data on agent self-awareness and response refinement
- **Debugging**: Identify when agents lose track of their identity or context

**Use cases:**
- Analyzing the effectiveness of self-reflection
- Studying agent self-awareness over time
- Identifying when agents drift from their intended behavior
- Measuring response quality improvements from refinement

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

### Explanation:
This feature modifies the main conversation loop to include additional initialization and tracking steps:

**New steps in the loop:**
1. **Initialize personality traits**: Call `initialize_personality_traits()` for both agents before conversation starts
2. **Question checks**: Call `question_check()` for both agents at the start of each turn
3. **Track initial/final responses**: Store both initial and final audience responses
4. **Removed learning()**: No longer calls the `learning()` method (reflections handled differently)

**Purpose:**
- **Proper initialization**: Ensures agents are properly set up with their personality profiles
- **Self-awareness tracking**: Monitors agent self-awareness throughout the conversation
- **Response refinement tracking**: Captures both initial and refined responses for analysis
- **Simplified flow**: Removes separate learning step (integrated into self-reflection)

**How it works:**
- **Before conversation**: Initialize personality traits for both agents
- **Each turn**:
  1. Run question checks (personality and context awareness)
  2. Actor acts (generates initial response, then refines it)
  3. Audience evaluates and responds (generates initial response, then refines it)
  4. Actor updates particle filter
  5. Store both initial and final responses in TurnLog

**Benefits:**
- More comprehensive logging and tracking
- Better initialization of agent personalities
- Continuous monitoring of agent self-awareness
- Complete record of response refinement process

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

### Explanation:
This feature changes how personality traits are assigned to agents in the main function. Instead of using **score-based traits** (numeric values 0-3), it uses **parametric traits** (text assertions).

**What changed:**
- **Before**: `generate_trait_scores()` returns `Dict[str, int]` with scores 0-3
- **After**: `generate_parametric_traits()` returns `List[str]` with trait assertion statements

**How it works:**
- **Audience agents**: Get positive trait assertions (e.g., "I tend to focus on individual parts...")
- **Actor agents**: Get negative trait assertions (e.g., "I do not tend to focus on individual parts...")
- These assertions are passed directly to agents as lists of strings

**Purpose:**
- **Explicit statements**: Agents receive explicit statements about their traits rather than numeric scores
- **Clearer distinction**: Makes the difference between audience and actor personalities more obvious
- **Better prompts**: LLMs may respond better to explicit statements than numeric scores
- **Parametric control**: Allows fine-grained control over trait expression

**Benefits:**
- More explicit trait representation
- Potentially better LLM understanding of traits
- Clearer distinction between agent types
- More flexible trait assignment

**Trade-offs:**
- Requires `negative_assertion` field in `PersonalityTrait` (see #2)
- More verbose than numeric scores
- May require more prompt space

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

### Explanation:
This feature (or bug) involves calling the particle filter's resampling method. The code calls `systematic_resample()` without an underscore, but the actual method name is `_systematic_resample()` (with underscore, indicating it's private).

**What it does:**
The `systematic_resample()` method performs **systematic resampling** of particles in the particle filter. This is used when particle diversity drops too low (measured by Effective Sample Size, ESS).

**How systematic resampling works:**
1. Generates evenly-spaced positions across [0, 1]
2. Maps these positions to cumulative weight distribution
3. Selects particles based on their weights
4. Returns indices of selected particles

**Purpose:**
- **Particle diversity**: Prevents particle degeneracy (all particles converging to same value)
- **Filter stability**: Maintains filter performance over many iterations
- **State tracking**: Ensures particle filter can track changing states effectively

**The issue:**
- Method is defined as `_systematic_resample()` (private, with underscore)
- Code calls it as `systematic_resample()` (public, without underscore)
- This would cause an `AttributeError` unless the method is actually public

**Possible explanations:**
1. **Bug**: The method should be called with underscore
2. **Different version**: The ParticleFilter class might have a public version
3. **Typo**: The method name might be incorrect in the code

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
- ✅ Self-reflection mechanism (`IMPESelfAssessmentComponent` - more sophisticated, **now for both actor and audience**)
- ✅ Two-stage response generation (via self-assessment component, **now for both actor and audience**)

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
