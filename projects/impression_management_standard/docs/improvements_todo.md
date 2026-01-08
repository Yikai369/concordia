# Improvements TODO

## Cultural Norms Initialization

**Issue**: The text from `initialize_cultural_norms()` should always be provided in every turn's prompts, not just sent once during initialization.

**Current Behavior**:
- `CulturalNormsComponent.initialize_norms()` sends a one-time initialization prompt with the full context: "You are in an alternative world in the year 3025 where there is a new set of cultural norms..."
- `get_norms_text()` only returns the norms list without the initialization context
- The initialization context is lost after the first call

**Problem**:
- LLMs don't have persistent memory between API calls
- The initialization context (alternative world, year 3025, consequences of not following norms) needs to be included in every prompt to maintain consistency
- Currently, only the norms list is included via `get_norms_text()`, but the important context about the alternative world is missing

**Solution**:
- Modify `get_norms_text()` to include the full initialization text, or
- Create a separate method that returns the full initialization text
- Ensure this full text is included in every prompt via `_get_prompt_header()` methods in components that use norms

**Reference**:
- Original implementation: `projects/impression_management/pe_conversation_openai.py` line 402
- Standard implementation: `concordia/components/agent/impression_management_pe.py` lines 272-287 (initialize_norms) and 262-270 (get_norms_text)

---

## Separate Observation and Action History

**Issue**: Currently, memory stores all utterances in a single `conversation` list, but doesn't explicitly separate observations (what the agent observed) from actions (what the agent said/did).

**Current Behavior**:
- `AgentMemory` / `IMPEMemoryComponent` has a single `conversation` list that stores all utterances
- When an agent speaks, the utterance is added to their memory
- When an agent observes, the observed utterance is also added to their memory
- Both are mixed together in the same list

**Problem**:
- No clear distinction between "what I observed" vs "what I said"
- Cannot easily retrieve:
  - All observations the agent made (what they saw/heard from others)
  - All actions the agent took (what they said/did)
- History presentation doesn't clearly show the agent's perspective (what they observed vs. what they did)

**Ideal Behavior**:
Each agent should maintain separate lists:
1. **Observation History**: What the agent observed each turn
   - Turn 1: Observed nothing (first turn)
   - Turn 2: Observed "Good approach" from Jane
   - Turn 3: Observed "I will demonstrate skills" from John
   - etc.

2. **Action History**: What the agent said/did each turn
   - Turn 1: Action: "I emphasize collaboration"
   - Turn 2: Action: "I will demonstrate technical skills"
   - Turn 3: Action: "My approach focuses on teamwork"
   - etc.

**Benefits**:
- Clear separation of observations vs. actions
- Can retrieve "all my observations" or "all my actions" separately
- Better for analysis and debugging
- More accurate representation of agent's perspective
- Can present history as: "I observed X, then I said Y, then I observed Z, then I said W..."

**Solution**:
- Add `observation_history: List[ObservationRecord]` to memory
- Add `action_history: List[ActionRecord]` to memory
- Create `ObservationRecord` dataclass with: `turn`, `observed_from`, `text`, `body`
- Create `ActionRecord` dataclass with: `turn`, `text`, `body`
- Update components to:
  - Store observations in `observation_history` when `pre_observe()` is called
  - Store actions in `action_history` when `act()` generates an utterance
- Add methods: `get_recent_observations(k)`, `get_recent_actions(k)`
- Update history presentation to show both observations and actions separately

**History Format Requirement**:
The observed history at every turn should be formatted as:
```
"At turn X, you observed Y, you did Z, and the outcome is T"
```

Where:
- **X**: Turn number
- **Y**: What the agent observed (partner's utterance, body language, etc.)
- **Z**: What the agent did (their own utterance/action)
- **T**: The outcome (evaluation I_t, belief I_hat, PE, etc.)

**Example Format**:
```
At turn 1, you observed nothing (first turn), you did "I emphasize collaboration" (body: "Maintains eye contact"), and the outcome is I_t=0.70, I_hat=0.50, PE=0.00.

At turn 2, you observed "Good approach" from Jane (body: "Nods"), you did "I will demonstrate technical skills" (body: "Confident posture"), and the outcome is I_t=0.80, I_hat=0.75, PE=+0.05.

At turn 3, you observed "The candidate shows potential" from Jane (body: "Maintains steady posture"), you did "My approach focuses on teamwork" (body: "Smiles"), and the outcome is I_t=0.85, I_hat=0.82, PE=-0.03.
```

**Implementation Notes**:
- Each turn's record should combine: observation + action + outcome
- Outcome should include: I_t (true evaluation), I_hat (belief), PE (prediction error)
- Format should be consistent and readable
- Can be used in prompts to give agents a clear understanding of their interaction history

**Reference**:
- Current memory structure: `concordia/components/agent/impression_management_pe.py` lines 150-174
- Utterance storage: `add_utterance()` method (line 168-174)
- Observation extraction: `IMPEAudienceEvaluationComponent.pre_observe()` (line 380-394)
- Action generation: `IMPEActComponent.get_action_attempt()` (line 781-858)
- PF update: `IMPEActorParticleFilterComponent.post_observe()` (line 540-634) - provides I_hat and PE
- Evaluation: `IMPEAudienceEvaluationComponent.post_observe()` (line 415-474) - provides I_t

---

## Thread Safety for IMPEMemoryComponent

**Issue**: `IMPEMemoryComponent` currently has no thread synchronization, but multiple components write to it in parallel during the same phase, which can cause race conditions.

**Current Behavior**:
- `IMPEMemoryComponent` stores data in simple Python lists (`_conversation`, `_pe_history`, `_reflections`, `_pf_history`, `_evaluation_history`)
- Multiple components write to memory simultaneously during parallel execution:
  - `IMPEAudienceEvaluationComponent.post_observe()` → calls `add_utterance()`, `add_evaluation_record()`
  - `IMPEActorParticleFilterComponent.post_observe()` → calls `update_particle_filter_state()`, `add_pe_record()`
  - `IMPEActComponent.get_action_attempt()` → calls `add_utterance()`
  - `IMPEReflectionComponent._make_pre_act_value()` → calls `add_reflection()`
- No locks or synchronization mechanisms are in place
- Methods like `update_particle_filter_state()` perform multiple operations (set particles, set weights, append to history) which are not atomic

**Problem**:
- **Race Conditions**: When multiple components write in parallel (e.g., during `POST_OBSERVE` phase), concurrent list modifications can cause:
  - Lost updates (one write overwrites another)
  - Inconsistent state (partial updates visible to other threads)
  - Data corruption (especially for multi-step operations like `update_particle_filter_state()`)
- **Non-Atomic Operations**: `update_particle_filter_state()` does three operations:
  ```python
  self._pf_particles = list(particles)      # Operation 1
  self._pf_weights = list(weights)          # Operation 2
  self._pf_history.append(history_entry)    # Operation 3
  ```
  If interrupted between operations, memory can be in an inconsistent state
- **Read-Write Conflicts**: Components reading from memory (e.g., `get_recent_conversation()`) while others are writing can see partial or inconsistent data

**Solution**:
Add thread locks to all memory access methods, similar to how `ListMemory` and `AssociativeMemory` handle concurrency.

**Implementation Steps**:

1. **Add threading lock to `__init__`**:
   ```python
   import threading

   def __init__(self, goal: Goal, recent_k: int = 3, pre_act_label: str = 'IMPE Memory'):
       super().__init__(goal=goal, recent_k=recent_k, pre_act_label=pre_act_label)
       self._lock = threading.Lock()  # Add this
       # ... rest of init ...
   ```

2. **Protect all write methods with locks**:
   - `add_utterance()` - wrap `self._conversation.append()` in lock
   - `add_evaluation_record()` - wrap `self._evaluation_history.append()` in lock
   - `update_particle_filter_state()` - wrap all three operations in lock
   - `add_pe_record()` (inherited from `PEMemoryComponent`) - check if parent needs lock too
   - `add_reflection()` (inherited from `PEMemoryComponent`) - check if parent needs lock too

3. **Protect read methods with locks** (for consistency):
   - `get_recent_conversation()` - wrap list slicing in lock
   - `get_recent_evaluations()` - wrap list slicing in lock
   - `get_pf_history()` - wrap list slicing in lock
   - `get_pf_state()` - wrap tuple creation in lock
   - All other getter methods that access internal lists

4. **Update `get_state()` and `set_state()`** (for checkpointing):
   - Wrap state serialization/deserialization in locks
   - Ensure atomic state snapshots

**Example Implementation**:
```python
def add_utterance(self, turn: int, speaker: str, text: str, body: str = '') -> None:
    """Add conversation utterance with body language."""
    with self._lock:
        self._conversation.append(
            Utterance(turn=turn, speaker=speaker, text=text, body=body)
        )

def update_particle_filter_state(
    self,
    particles: list[float],
    weights: list[float],
    history_entry: dict[str, Any],
) -> None:
    """Update particle filter state."""
    with self._lock:
        self._pf_particles = list(particles)
        self._pf_weights = list(weights)
        self._pf_history.append(history_entry)

def get_recent_conversation(self, k: int | None = None) -> list[Utterance]:
    """Get recent conversation entries."""
    if k is None:
        k = self._recent_k
    with self._lock:
        return self._conversation[-k:].copy()  # Return copy to avoid holding lock
```

**Considerations**:
- **Performance**: Locks add minimal overhead for this use case (conversation data, not high-frequency operations)
- **Deadlocks**: Use a single lock for all operations to avoid deadlock risk
- **Return Copies**: For getter methods, return copies of lists (`.copy()`) to avoid holding the lock while caller processes data
- **Parent Class**: Check if `PEMemoryComponent` (parent class) also needs locks for `add_pe_record()` and `add_reflection()`

**Benefits**:
- **Thread Safety**: Prevents race conditions during parallel component execution
- **Data Integrity**: Ensures memory state is always consistent
- **Atomic Operations**: Multi-step operations (like PF update) are now atomic
- **Framework Compatibility**: Aligns with Concordia's design patterns (ListMemory, AssociativeMemory use locks)

**Reference**:
- Current implementation: `concordia/components/agent/impression_management_pe.py` lines 150-244
- Thread-safe example: `concordia/components/agent/memory.py` lines 238-338 (ListMemory with locks)
- Components that write in parallel:
  - `IMPEAudienceEvaluationComponent.post_observe()` (line 415-474)
  - `IMPEActorParticleFilterComponent.post_observe()` (line 540-634)
  - `IMPEActComponent.get_action_attempt()` (line 781-858)
  - `IMPEReflectionComponent._make_pre_act_value()` (line 665-750)

---

## Information Flow History Bank

**Issue**: Currently, there is no comprehensive system to capture and persist all model inputs (prompts) and outputs (responses) for each agent. This makes debugging difficult when trying to understand what prompts led to specific agent behaviors.

**Current Behavior**:
- `EntityAgentWithLogging` uses `Measurements` object to collect component logs
- Components can implement `ComponentWithLogging` interface
- Some components log prompts (e.g., `ScriptedActComponent`)
- Logs are stored in-memory only via `Measurements.get_all_channels()`
- No centralized, persistent history bank that captures all LLM interactions

**Problem**:
- **Not Comprehensive**: Not all components log full prompts/responses
- **Not Persistent**: Logs are in-memory, lost after simulation ends
- **Not Centralized**: Each component logs separately, no unified view per agent
- **No Model-Level Interception**: Can't capture all LLM calls automatically
- **Debugging Difficulty**: Cannot trace exactly what prompts led to specific responses
- **No Reproducibility**: Cannot replay or analyze model interactions after simulation

**Ideal Behavior**:
Each agent should have a complete history bank that stores:
1. **All Model Interactions**: Every call to `sample_text()` and `sample_choice()`
2. **Full Context**: Complete prompts sent to the model (not just summaries)
3. **Complete Responses**: Full model responses
4. **Metadata**: Turn number, component name, phase (pre_act/act/post_act/observe), method parameters
5. **Persistent Storage**: Save to JSON for later analysis
6. **Per-Agent Separation**: Each agent's history is stored separately

**Benefits**:
- **Complete Visibility**: See every model call made by each agent
- **Debugging**: Trace exactly what prompts led to specific responses
- **Reproducibility**: Save and replay model interactions
- **Analysis**: Analyze prompt patterns, response quality, token usage, etc.
- **Non-Intrusive**: Works with existing code via wrapper pattern (opt-in feature)

**Solution**:
Implement a model-level interception system that:
1. Wraps language models with a logging wrapper that captures all calls
2. Stores interactions in a persistent history bank per agent
3. Tracks component context, phase, and turn information
4. Provides save/load functionality for analysis
5. Integrates seamlessly with existing simulation architecture

**Implementation Plan**:
A comprehensive implementation plan has been created with detailed steps, code examples, and integration points.

**Reference**:
- Implementation plan: `plans/information_flow_history_bank_plan.md`
- Current logging infrastructure: `concordia/agents/entity_agent_with_logging.py`
- Component logging example: `concordia/components/agent/scripted_act.py` (lines 141-151)
- Model interface: `concordia/language_model/language_model.py`

---

## Self-Assessment Component

**Issue**: Currently, there is no mechanism to ensure that agent responses align with their background information (personality traits, cultural norms, goals, and context). Agents may generate responses that are inconsistent with their stated characteristics.

**Current Behavior**:
- `IMPEActComponent` generates responses based on prompts that include personality traits, cultural norms, and goals
- No validation or quality control checks if the generated response actually aligns with the background information
- Responses are returned directly without consistency assessment
- No feedback mechanism to improve alignment

**Problem**:
- **Inconsistency**: Agents may generate responses that contradict their personality traits or violate cultural norms
- **No Quality Control**: No mechanism to catch inconsistencies before they enter the conversation
- **No Feedback Loop**: Agents don't receive feedback on whether their responses align with their background
- **Debugging Difficulty**: Hard to identify when/why agents generate inconsistent responses
- **Unreliable Behavior**: Agent behavior may drift from intended characteristics over time

**Ideal Behavior**:
A self-assessment component should:
1. **Assess Consistency**: Evaluate whether generated responses align with personality traits, cultural norms, goals, and context
2. **Provide Feedback**: Generate feedback on what is inconsistent and how to improve
3. **Optionally Revise**: Generate revised responses when inconsistencies are detected
4. **Log Assessments**: Track consistency scores and revision history for analysis
5. **Be Optional**: Can be enabled/disabled without affecting other components

**Solution**:
Implement `IMPESelfAssessmentComponent` as a wrapper around `IMPEActComponent` that:
1. **Wraps Base Component**: Intercepts responses from `IMPEActComponent` before they are returned
2. **Assesses Consistency**: Uses LLM to evaluate consistency between response and background information
   - Rates consistency on scale 0.0-1.0
   - Provides feedback on inconsistencies
   - Determines if revision is needed based on threshold
3. **Revises if Necessary**: When consistency < threshold and revision enabled:
   - Uses feedback to generate revised response
   - Ensures revised response maintains core message while better aligning with background
4. **Handles Memory**: Updates memory with final (possibly revised) utterance, removing original if revised
5. **Logs Results**: Records consistency scores, feedback, and revision status

**Component Design**:
- **Type**: `ActingComponent` (wrapper pattern)
- **Location**: `concordia/components/agent/impression_management_pe.py`
- **Parameters**:
  - `base_act_component`: The `IMPEActComponent` instance to wrap
  - `consistency_threshold`: Minimum consistency score (0-1) to accept without revision (default: 0.7)
  - `enable_revision`: Whether to revise when inconsistent (default: True)
- **Integration**: Optional component controlled via prefab parameters

**Implementation Steps**:
1. Create `IMPESelfAssessmentComponent` class
2. Implement `get_action_attempt()` method that:
   - Calls base component to generate initial response
   - Collects context (norms, traits, goal, I_hat, reflections, conversation)
   - Assesses consistency via LLM prompt
   - Revises if needed (when consistency < threshold)
   - Updates memory with final utterance
   - Returns final response
3. Add `skip_memory_update` parameter to `IMPEActComponent.get_action_attempt()` to prevent duplicate memory updates
4. Update actor prefab to optionally wrap `IMPEActComponent` with self-assessment
5. Add command-line arguments: `--enable_self_assessment`, `--consistency_threshold`, `--disable_revision`
6. (Optional) Add `AssessmentRecord` to `IMPEMemoryComponent` to track assessment history

**Benefits**:
- **Consistency Enforcement**: Ensures agent behavior aligns with stated traits/norms
- **Quality Control**: Catches inconsistencies before they enter conversation
- **Adaptive Behavior**: Agent learns to generate more consistent responses over time
- **Debugging**: Assessment records help identify when/why inconsistencies occur
- **Modularity**: Can be enabled/disabled without changing other components
- **Backward Compatible**: Default behavior unchanged (self-assessment disabled by default)

**Considerations**:
- **Performance**: Adds LLM calls, increasing latency (can use faster model for assessment)
- **Over-Correction**: Revision might change meaning too much (mitigated by "maintain core message" constraint)
- **Assessment Quality**: LLM might not assess accurately (use structured output format, examples)
- **Memory Handling**: Need to prevent duplicate utterances when revising (use `skip_memory_update` flag)

**Reference**:
- Implementation plan: `projects/impression_management_standard/docs/self_assessment_component_plan.md`
- Base component: `concordia/components/agent/impression_management_pe.py` - `IMPEActComponent` (line 781-858)
- Actor prefab: `concordia/prefabs/entity/impression_management_actor.py`
- Memory component: `concordia/components/agent/impression_management_pe.py` - `IMPEMemoryComponent` (line 150-244)

---

## Agent Identity and Self-Awareness Questions

**Issue**: Agents in the impression_management_standard project are not asked the standard identity and self-awareness questions that are commonly used in Concordia framework examples. This may reduce agent self-awareness and consistency.

**Current Behavior**:
- Agents receive goal information, cultural norms, personality traits, conversation history, and reflections
- Agents are told "You are {name}" but are not explicitly asked to reflect on their identity
- No explicit questions about "who you are", "what kind of person you are", or "what situation you are in"
- No role-playing instructions component that explains the experimental context

**Standard Questions in Examples**:
Examples in the Concordia framework commonly use these components:

1. **Instructions Component** (`concordia/components/agent/instructions.py`):
   - Provides role-playing instructions: "This is a social science experiment... play the role of {agent_name} as accurately as possible... Always use third-person limited perspective."
   - Explains the experimental context and expectations
   - Helps agents understand they are playing a character in a simulation

2. **SelfPerception Component** (`concordia/components/agent/question_of_recent_memories.py`):
   - Asks: "What kind of person is {agent_name}?"
   - Helps agents develop and maintain a consistent self-concept
   - Answers are generated based on recent memories and observations

3. **SituationPerception Component**:
   - Asks: "What kind of situation is {agent_name} in right now?"
   - Helps agents understand their current context
   - Answers consider observations, somatic state, and relevant memories

4. **PersonBySituation Component**:
   - Asks: "What would a person like {agent_name} do in a situation like this?"
   - Helps agents reason about appropriate actions based on their identity and situation
   - Combines self-perception and situation perception

**Problem**:
- **Reduced Self-Awareness**: Agents may not develop a clear sense of their identity without explicit self-perception questions
- **Inconsistent Behavior**: Without asking "who am I?", agents may not maintain consistent character traits
- **Missing Context**: Lack of Instructions component means agents don't understand the experimental/simulation context
- **Less Grounded Actions**: Without situation perception, agents may not fully understand their current context
- **Framework Misalignment**: Not following standard Concordia patterns used in examples

**Ideal Behavior**:
Agents should be asked (via components that provide context in `pre_act`):
1. **Role-Playing Instructions**: Understand they are in a social science experiment, playing a character
2. **Self-Perception**: "What kind of person am I?" - based on traits, norms, and recent behavior
3. **Situation Perception**: "What kind of situation am I in right now?" - based on observations and context
4. **Person-by-Situation**: "What would a person like me do in this situation?" - guides action selection

**Solution**:
Add standard Concordia identity and self-awareness components to the actor prefab:

1. **Add Instructions Component**:
   ```python
   from concordia.components.agent import instructions

   instructions_comp = instructions.Instructions(
       agent_name=entity_name,
       pre_act_label='\nRole playing instructions',
   )
   ```

2. **Add SelfPerception Component** (optional, but recommended):
   ```python
   from concordia.components.agent.question_of_recent_memories import SelfPerception

   self_perception = SelfPerception(
       model=model,
       pre_act_label=f'\nQuestion: What kind of person is {entity_name}?\nAnswer',
   )
   ```

3. **Add SituationPerception Component** (optional):
   ```python
   from concordia.components.agent.question_of_recent_memories import SituationPerception

   situation_perception = SituationPerception(
       model=model,
       components={
           'Observation': observation_component,
           # ... other context components
       },
       pre_act_label=f'\nQuestion: What kind of situation is {entity_name} in right now?\nAnswer',
   )
   ```

4. **Add PersonBySituation Component** (optional):
   ```python
   from concordia.components.agent.question_of_recent_memories import PersonBySituation

   person_by_situation = PersonBySituation(
       model=model,
       components={
           'SelfPerception': self_perception,
           'SituationPerception': situation_perception,
       },
       pre_act_label=f'\nQuestion: What would a person like {entity_name} do in a situation like this?\nAnswer',
   )
   ```

**Implementation Considerations**:
- **Instructions Component**: Should be added as a high-priority component (provides essential context)
- **SelfPerception**: Can use existing memory and traits to answer "who am I?"
- **SituationPerception**: Can use observations and conversation history
- **PersonBySituation**: Combines self-perception and situation to guide actions
- **Integration**: These components provide context in `pre_act`, which is automatically included in action prompts
- **Optional vs Required**: Instructions should be required; others can be optional via parameters

**Benefits**:
- **Better Self-Awareness**: Agents develop clearer sense of identity
- **More Consistent Behavior**: Self-perception helps maintain character consistency
- **Better Context Understanding**: Situation perception helps agents understand their environment
- **Framework Alignment**: Follows standard Concordia patterns
- **Improved Action Quality**: Person-by-situation reasoning can improve action selection

**Reference**:
- Instructions component: `concordia/components/agent/instructions.py`
- SelfPerception component: `concordia/components/agent/question_of_recent_memories.py` (line 210)
- Example usage: `examples/deprecated/modular/environment/supporting_agent_factory/basic_agent.py` (lines 115-155)
- Example usage: `examples/selling_cookies.ipynb` (lines 233-255)
- Current actor prefab: `concordia/prefabs/entity/impression_management_actor.py`

---

## Framework Integration Opportunities

**Overview**: Several opportunities exist to better utilize existing Concordia framework components. These would improve code reuse, maintainability, and alignment with framework patterns.

**Detailed Analysis**: See `framework_integration_opportunities.md` for comprehensive review of Concordia framework features and detailed implementation guidance.

### High Priority Opportunities

**1. Better Observation Component Usage**
- **Status**: ⚠️ Partially Implemented
- **Current**: Using `ObservationToMemory` but not fully leveraging observation retrieval components
- **Opportunity**: Use `LastNObservations` or `ObservationsSinceLastPreAct` for better observation retrieval
- **Benefits**: Consistent framework patterns, automatic observation management, better separation of concerns
- **Details**: See `framework_integration_opportunities.md` - Opportunity 1

**2. Leverage AssociativeMemory for Semantic Search**
- **Status**: ⚠️ Partially Implemented
- **Current**: `AssociativeMemory` exists but not fully utilized for semantic search
- **Opportunity**: Use `AllSimilarMemories` component for semantic retrieval of relevant past conversations
- **Benefits**: Semantic similarity search, find relevant conversations by meaning, better context retrieval
- **Details**: See `framework_integration_opportunities.md` - Opportunity 2

**3. Add Thread Safety to IMPEMemoryComponent**
- **Status**: ❌ Not Implemented
- **Current**: No thread synchronization despite parallel component writes
- **Opportunity**: Add threading locks following `ListMemory`/`AssociativeMemory` pattern
- **Benefits**: Prevents race conditions, ensures data integrity, atomic operations
- **Details**: See "Thread Safety for IMPEMemoryComponent" section above and `framework_integration_opportunities.md` Part 4

### Medium Priority Opportunities

**4. Use QuestionOfRecentMemories for Context Queries**
- **Status**: ❌ Not Implemented
- **Current**: Manual conversation history formatting, fixed K-window retrieval
- **Opportunity**: Use `QuestionOfRecentMemories` for natural language memory queries
- **Benefits**: Natural language queries, context-aware retrieval, LLM-powered relevance ranking
- **Details**: See `framework_integration_opportunities.md` - Opportunity 3

**5. Evaluate ConcatActComponent for Action Composition**
- **Status**: ❌ Not Implemented
- **Current**: Direct action generation in `IMPEActComponent`, manual context aggregation
- **Opportunity**: Use `ConcatActComponent` to combine multiple component outputs
- **Benefits**: Better component separation, flexible action composition, easier to extend
- **Details**: See `framework_integration_opportunities.md` - Opportunity 4

### Low Priority Opportunities

**6. Consider ListMemory for General Observations**
- **Status**: ⚠️ Could Consider
- **Current**: Using `AssociativeMemory` for general observations
- **Opportunity**: Evaluate if `ListMemory` would be sufficient (lighter weight, no embeddings)
- **Benefits**: Lighter weight, still thread-safe, good for simple text observations
- **Details**: See `framework_integration_opportunities.md` - Opportunity 6

**7. Better Integration with Game Master Components**
- **Status**: ⚠️ Could Consider
- **Current**: Custom game master with manual coordination
- **Opportunity**: Leverage more standard game master components (`MakeObservation`, `NextActing`, `Terminate`, `WorldState`)
- **Benefits**: Standard patterns, reusable components, framework-maintained logic
- **Details**: See `framework_integration_opportunities.md` - Opportunity 7

### Summary

**Framework Alignment Status**:
- ✅ Strong: Component architecture, phase-based lifecycle, prefab system, standard loop, base class inheritance
- ⚠️ Partial: Observation components, memory capabilities, thread safety, action composition

**Key Takeaway**: The project is well-integrated but could better leverage framework-provided capabilities for observation management, semantic memory, and memory queries. See `framework_integration_opportunities.md` for detailed analysis and implementation guidance.
