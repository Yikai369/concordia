# Component Architecture: Actor and Audience Agents

This document describes the components that make up the Actor and Audience agents in the Impression Management PE (Prediction Error) conversation system. Components are listed in execution order from observation to action.

## Execution Flow Overview

The Concordia framework executes components in phases:

1. **Observation Phase** (when `observe()` is called):
   - `pre_observe()`: Components process incoming observation
   - `post_observe()`: Components react to observation
   - `update()`: Components update internal state

2. **Action Phase** (when `act()` is called):
   - `pre_act()`: Context components provide information to the LLM
   - `act()`: Acting component generates the action/utterance
   - `post_act()`: Components process the generated action
   - `update()`: Components update internal state

---

## Actor Agent Components

The Actor agent adapts its responses based on particle filter belief tracking of the audience's evaluation.

### Component Execution Order

#### Phase: Pre-Act (Context Components - provide information to LLM)

1. **Instructions** (Optional, default: enabled)
   - **Key**: `Instructions`
   - **Type**: `instructions.Instructions`
   - **Phase**: `pre_act`
   - **Purpose**: Provides role-playing instructions and experimental context
   - **Output**: Role-playing instructions for the agent

2. **SelfPerception** (Optional, default: enabled)
   - **Key**: `SelfPerception`
   - **Type**: `question_of_recent_memories.SelfPerception`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What kind of person is {agent_name}?"
   - **Output**: Self-description based on memories, traits, and norms

3. **SituationPerception** (Optional, default: disabled)
   - **Key**: `SituationPerception`
   - **Type**: `question_of_recent_memories.SituationPerception`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What kind of situation is {agent_name} in right now?"
   - **Output**: Situation description based on recent observations
   - **Note**: Required for PersonBySituation component

4. **PersonBySituation** (Optional, default: disabled)
   - **Key**: `PersonBySituation`
   - **Type**: `question_of_recent_memories.PersonBySituation`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What would a person like {agent_name} do in a situation like this?"
   - **Dependencies**: Requires SelfPerception and SituationPerception
   - **Output**: Behavioral prediction based on self and situation

5. **AssociativeMemory** (Required)
   - **Key**: `Memory`
   - **Type**: `agent_components.memory.AssociativeMemory`
   - **Phase**: `pre_act`, `pre_observe`, `update`
   - **Purpose**: Stores general observations and memories
   - **Output**: Recent memories for context

6. **IMPEMemoryComponent** (Required)
   - **Key**: `IMPE_Memory`
   - **Type**: `impression_management_pe.IMPEMemoryComponent`
   - **Phase**: `pre_act`, `pre_observe`, `update`
   - **Purpose**: Stores conversation history, PE records, reflections, and particle filter state
   - **Output**: Recent conversation, I_hat history, reflections

7. **IMPEActorParticleFilterComponent** (Required)
   - **Key**: `IMPE_ActorParticleFilter`
   - **Type**: `impression_management_pe.IMPEActorParticleFilterComponent`
   - **Phase**: `pre_act`
   - **Purpose**: Updates particle filter belief about audience's evaluation (I_hat)
   - **Output**: Current I_hat value and particle filter statistics

8. **IMPEReflectionComponent** (Required)
   - **Key**: `IMPE_Reflection`
   - **Type**: `impression_management_pe.IMPEReflectionComponent`
   - **Phase**: `pre_act`
   - **Purpose**: Generates reflection on how to improve goal achievement
   - **Output**: Reflection text based on current I_hat

9. **WorldContextComponent** (Optional, default: enabled)
   - **Key**: `WorldContext`
   - **Type**: `impression_management_pe.WorldContextComponent`
   - **Phase**: `pre_act`
   - **Purpose**: Provides 2A25 world-building context (Cadens, Riffers) and interview context
   - **Output**: World-building narrative and interview setting description
   - **Note**: Only includes world-building if agent has cultural norms

10. **CulturalNormsComponent** (Optional)
    - **Key**: `CulturalNorms`
    - **Type**: `impression_management_pe.CulturalNormsComponent`
    - **Phase**: `pre_act`
    - **Purpose**: Provides cultural norms that the agent must follow
    - **Output**: List of cultural norms with descriptions
    - **Note**: Typically not used for actor (only audience has norms)

11. **PersonalityTraitsComponent** (Optional)
    - **Key**: `PersonalityTraits`
    - **Type**: `impression_management_pe.PersonalityTraitsComponent`
    - **Phase**: `pre_act`
    - **Purpose**: Provides personality traits with scores
    - **Output**: List of personality traits with assertions

#### Phase: Pre-Observe (Observation Processing)

12. **ObservationToMemory** (Required)
    - **Key**: `ObservationToMemory`
    - **Type**: `agent_components.observation.ObservationToMemory`
    - **Phase**: `pre_observe`
    - **Purpose**: Stores observations in standard memory
    - **Output**: Stores observation in AssociativeMemory

#### Phase: Act (Action Generation)

13. **IMPEActComponent** (Required) - Base acting component
    - **Type**: `impression_management_pe.IMPEActComponent`
    - **Phase**: `act` (via `get_action_attempt()`)
    - **Purpose**: Generates utterance based on:
      - Goal and current belief (I_hat)
      - Recent conversation history
      - Recent I_hat history
      - Recent reflections
      - Cultural norms and personality traits (from pre_act context)
      - World context (from pre_act context)
    - **Output**: Formatted action string: `"DIALOGUE: <text>\nBODY: <body>"`

14. **IMPESelfAssessmentComponent** (Optional, wraps IMPEActComponent)
    - **Type**: `impression_management_pe.IMPESelfAssessmentComponent`
    - **Phase**: `act` (wraps base act component)
    - **Purpose**:
      - Assesses consistency of generated response with traits, norms, and goals
      - Optionally revises response if consistency is below threshold
    - **Output**: Original or revised action string
    - **Note**: Only active if `enable_self_assessment=True`

---

## Audience Agent Components

The Audience agent evaluates the actor's performance and provides feedback.

### Component Execution Order

#### Phase: Pre-Act (Context Components - provide information to LLM)

1. **Instructions** (Optional, default: enabled)
   - **Key**: `Instructions`
   - **Type**: `instructions.Instructions`
   - **Phase**: `pre_act`
   - **Purpose**: Provides role-playing instructions and experimental context
   - **Output**: Role-playing instructions for the agent

2. **SelfPerception** (Optional, default: enabled)
   - **Key**: `SelfPerception`
   - **Type**: `question_of_recent_memories.SelfPerception`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What kind of person is {agent_name}?"
   - **Output**: Self-description based on memories, traits, and norms

3. **SituationPerception** (Optional, default: disabled)
   - **Key**: `SituationPerception`
   - **Type**: `question_of_recent_memories.SituationPerception`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What kind of situation is {agent_name} in right now?"
   - **Output**: Situation description based on recent observations
   - **Note**: Required for PersonBySituation component

4. **PersonBySituation** (Optional, default: disabled)
   - **Key**: `PersonBySituation`
   - **Type**: `question_of_recent_memories.PersonBySituation`
   - **Phase**: `pre_act`
   - **Purpose**: Answers "What would a person like {agent_name} do in a situation like this?"
   - **Dependencies**: Requires SelfPerception and SituationPerception
   - **Output**: Behavioral prediction based on self and situation

5. **AssociativeMemory** (Required)
   - **Key**: `Memory`
   - **Type**: `agent_components.memory.AssociativeMemory`
   - **Phase**: `pre_act`, `pre_observe`, `update`
   - **Purpose**: Stores general observations and memories
   - **Output**: Recent memories for context

6. **IMPEMemoryComponent** (Required)
   - **Key**: `IMPE_Memory`
   - **Type**: `impression_management_pe.IMPEMemoryComponent`
   - **Phase**: `pre_act`, `pre_observe`, `update`
   - **Purpose**: Stores conversation history and evaluation records
   - **Output**: Recent conversation and evaluation history

#### Phase: Pre-Observe (Observation Processing)

7. **ObservationToMemory** (Required)
   - **Key**: `ObservationToMemory`
   - **Type**: `agent_components.observation.ObservationToMemory`
   - **Phase**: `pre_observe`
   - **Purpose**: Stores observations in standard memory
   - **Output**: Stores observation in AssociativeMemory

8. **IMPEAudienceEvaluationComponent** (Required)
   - **Key**: `IMPE_AudienceEvaluation`
   - **Type**: `impression_management_pe.IMPEAudienceEvaluationComponent`
   - **Phase**:
     - `pre_observe`: Extracts actor's utterance from observation
     - `post_observe`: Evaluates actor and generates response
   - **Purpose**:
     - Extracts actor's text and body language from observation
     - Evaluates actor's competence (I_t) on scale [0,1]
     - Generates feedback response matching the evaluation score
   - **Output**:
     - Stores I_t and evaluation utterance in IMPEMemoryComponent
     - Returns evaluation result string

#### Phase: Pre-Act (Context Components - continued)

9. **WorldContextComponent** (Optional, default: enabled)
   - **Key**: `WorldContext`
   - **Type**: `impression_management_pe.WorldContextComponent`
   - **Phase**: `pre_act`
   - **Purpose**: Provides 2A25 world-building context (Cadens, Riffers) and interview context
   - **Output**: World-building narrative and interview setting description
   - **Note**: Only includes world-building if agent has cultural norms

10. **CulturalNormsComponent** (Optional, typically enabled for audience)
    - **Key**: `CulturalNorms`
    - **Type**: `impression_management_pe.CulturalNormsComponent`
    - **Phase**: `pre_act`
    - **Purpose**: Provides cultural norms that the agent must follow
    - **Output**: List of cultural norms with descriptions
    - **Note**: Typically enabled for audience (interviewer) to evaluate based on norms

11. **PersonalityTraitsComponent** (Optional)
    - **Key**: `PersonalityTraits`
    - **Type**: `impression_management_pe.PersonalityTraitsComponent`
    - **Phase**: `pre_act`
    - **Purpose**: Provides personality traits with scores
    - **Output**: List of personality traits with assertions

#### Phase: Act (Action Generation)

12. **SimpleAudienceActComponent** (Required) - Base acting component
    - **Type**: `audience_act_component.SimpleAudienceActComponent`
    - **Phase**: `act` (via `get_action_attempt()`)
    - **Purpose**: Retrieves and returns the stored evaluation response from IMPEMemoryComponent
    - **Output**: Formatted action string: `"DIALOGUE: <text>\nBODY: <body>"`
    - **Note**: The actual response was generated by IMPEAudienceEvaluationComponent during `post_observe`

13. **IMPESelfAssessmentComponent** (Optional, wraps SimpleAudienceActComponent)
    - **Type**: `impression_management_pe.IMPESelfAssessmentComponent`
    - **Phase**: `act` (wraps base act component)
    - **Purpose**:
      - Assesses consistency of generated response with traits, norms, and goals
      - Optionally revises response if consistency is below threshold
    - **Output**: Original or revised action string
    - **Note**: Only active if `enable_self_assessment=True`

---

## Component Interaction Flow

### Actor Turn Flow

```
1. Actor.act() called
   ↓
2. PRE_ACT phase: All context components provide information
   - Instructions, SelfPerception, SituationPerception, PersonBySituation
   - IMPEMemoryComponent (conversation history, I_hat, reflections)
   - IMPEActorParticleFilterComponent (updates I_hat)
   - IMPEReflectionComponent (generates reflection)
   - WorldContextComponent, CulturalNormsComponent, PersonalityTraitsComponent
   ↓
3. ACT phase: IMPEActComponent (or IMPESelfAssessmentComponent wrapper)
   - Uses all pre_act context to generate utterance
   - Stores utterance in IMPEMemoryComponent
   ↓
4. POST_ACT phase: Components process the action
   ↓
5. UPDATE phase: Components update state
```

### Audience Turn Flow

```
1. Audience.observe(actor_utterance) called
   ↓
2. PRE_OBSERVE phase:
   - ObservationToMemory: Stores observation in standard memory
   - IMPEAudienceEvaluationComponent: Extracts actor's text/body language
   ↓
3. POST_OBSERVE phase:
   - IMPEAudienceEvaluationComponent:
     * Evaluates actor (I_t)
     * Generates feedback response
     * Stores both in IMPEMemoryComponent
   ↓
4. UPDATE phase: Components update state
   ↓
5. Audience.act() called (next step in simulation)
   ↓
6. PRE_ACT phase: Context components provide information
   - Instructions, SelfPerception, etc.
   - WorldContextComponent, CulturalNormsComponent, PersonalityTraitsComponent
   ↓
7. ACT phase: SimpleAudienceActComponent (or IMPESelfAssessmentComponent wrapper)
   - Retrieves stored evaluation response from IMPEMemoryComponent
   - Returns it as action
   ↓
8. POST_ACT phase: Components process the action
   ↓
9. UPDATE phase: Components update state
```

---

## Key Differences: Actor vs Audience

| Aspect | Actor | Audience |
|--------|-------|----------|
| **Particle Filter** | ✅ Has IMPEActorParticleFilterComponent | ❌ No particle filter |
| **Belief Tracking** | Tracks I_hat (estimated audience evaluation) | Uses I_t (true evaluation score) |
| **Reflection** | ✅ Has IMPEReflectionComponent | ❌ No reflection component |
| **Evaluation** | ❌ Does not evaluate | ✅ Has IMPEAudienceEvaluationComponent |
| **Cultural Norms** | ❌ Typically none | ✅ Typically enabled |
| **Act Component** | IMPEActComponent (generates based on I_hat) | SimpleAudienceActComponent (returns stored response) |
| **Action Generation** | Generates new utterance each turn | Returns pre-generated evaluation response |

---

## Component Dependencies

### Required Dependencies
- **PersonBySituation** requires: SelfPerception AND SituationPerception
- **All components** require: IMPEMemoryComponent (for storing/retrieving data)
- **Act components** require: All pre_act context components (for prompt generation)

### Optional Enhancements
- **IMPESelfAssessmentComponent**: Can wrap any ActingComponent
- **WorldContextComponent**: Only includes world-building if CulturalNormsComponent is present
- **CulturalNormsComponent**: Can be initialized once via `initialize_norms()`

---

## Configuration Flags

### Actor Configuration
- `enable_instructions`: Enable Instructions component (default: True)
- `enable_self_perception`: Enable SelfPerception component (default: True)
- `enable_situation_perception`: Enable SituationPerception component (default: False)
- `enable_person_by_situation`: Enable PersonBySituation component (default: False)
- `enable_world_building`: Enable world-building context (default: True)
- `enable_interview_context`: Enable interview context (default: True)
- `enable_self_assessment`: Enable self-assessment wrapper (default: False)
- `consistency_threshold`: Consistency threshold for self-assessment (default: 0.7)
- `disable_revision`: Disable revision in self-assessment (default: False)

### Audience Configuration
- Same as Actor, plus:
- `cultural_norms`: List of CulturalNorm objects (typically provided)
- `traits`: List of PersonalityTrait objects (optional)
- `trait_scores`: Dictionary mapping trait names to scores (optional)

---

## Notes

1. **Component Order Matters**: Components are executed in the order they are added to `components_of_agent` dictionary. Dependencies should be added before components that depend on them.

2. **Pre-Act Context**: All `pre_act` components provide context that is aggregated and passed to the acting component. The acting component uses this context to build its prompt.

3. **Memory Updates**: Components update memory during `update()` phase to ensure consistency.

4. **Self-Assessment**: When enabled, `IMPESelfAssessmentComponent` wraps the base act component and can revise responses for consistency with traits, norms, and goals.

5. **World Context**: The world-building narrative is only included if the agent has cultural norms, ensuring it's contextually appropriate.




