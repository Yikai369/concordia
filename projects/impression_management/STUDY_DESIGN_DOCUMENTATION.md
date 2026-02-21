# Prediction Error-Driven Impression Management Study: Design Documentation

## Overview

This study simulates a dyadic conversation between two LLM-powered agents: an **Actor** (interviewee) and an **Audience** (interviewer). The Actor attempts to manage the impression they create by tracking their belief about how the Audience perceives them, using a particle filter to maintain probabilistic beliefs about the Audience's hidden evaluation state. The study measures how prediction errors (discrepancies between expected and actual feedback) drive adaptive behavior.

### Core Research Question

How do agents adapt their behavior when they experience prediction errors about how they are being perceived? The study operationalizes this through:
- **Prediction Error (PE)**: The difference between the Actor's prior belief and posterior belief about the Audience's evaluation
- **Learning Dynamics**: How the Actor adjusts their strategy based on PE signals
- **Belief Tracking**: A particle filter maintains a distribution over the Audience's true evaluation state

---

## Agent Roles and Interaction Flow

### The Actor (Interviewee)

**Goal**: Be perceived as highly competent (target: 1.0 on a 0-1 scale)

**What the Actor Observes**:

1. **World Context** (if enabled):
   - Interview setting: "an interview room in a corporate office"
   - Role mention: "the role of product manager"
   - **Minimal world description**:
     - "This is a Caden-majority institutional setting"
     - "If the interviewee is suspected to be a Riffer, they will be denied this job position"
     - "This scenario occurs inside the fictional world of 2A25"
   - The Actor sees that Cadens are the majority and Riffers are stigmatized, but does NOT see the detailed world-building explanation
   - Note: Detailed role requirements (responsibilities, evaluation criteria) are defined but not explicitly included in prompts

2. **Personality Traits**:
   - Their own behavioral profile (e.g., detail-focused, avoids eye contact)
   - Initialized separately via a dedicated prompt before conversation starts
   - Traits shape communication style throughout the interaction

3. **Goal Information**:
   - Goal name: "competence"
   - Goal description: Instructions on what they want to achieve
   - Goal ideal value: 1.0 (target score)

4. **Conversation History**:
   - Recent turns (last K turns, default K=3) of dialogue and body language
   - Includes both Actor's and Audience's utterances

5. **Audience Responses**:
   - The Audience's verbal responses (dialogue)
   - The Audience's body language descriptions

6. **Belief History**:
   - Recent I_hat values (belief estimates over time)
   - Used to track whether previous actions had intended effects

7. **Current Belief State**:
   - Current I_hat value (posterior mean belief about Audience's evaluation)
   - Used to condition actions on perceived performance

**What the Actor Does NOT Explicitly Observe**:
- **Cultural Norms**: The Actor does NOT have explicit knowledge of the 18 cultural norms that govern the Audience's behavior. They must infer appropriate behavior from feedback and context.
- **Detailed 2A25 World Description**: The Actor sees minimal world context (mentions Cadens, Riffers, 2A25) but does NOT see the full world-building explanation that describes:
  - Cadens as "the dominant cognitive-cultural order"
  - Riffers as "a social minority group with unique cultural knowledge and individual traits"
  - The detailed explanation of how Riffers need to adopt Caden norms to avoid social exclusion
  - The full world-building context (only the Audience sees this via cultural norms)
- **Detailed Role Requirements**: While the role is mentioned, specific responsibilities and evaluation criteria from `goal.role` are not explicitly included in prompts
- **Reflections**: While the Actor generates reflections, they are stored in memory but not explicitly included in subsequent prompts (only mentioned as something to "consider")

**What the Actor Does**:

1. **Acts** (Turn t):
   - Generates an utterance aimed at demonstrating competence
   - Includes both dialogue and body language
   - On first turn: acts based on goal alone
   - On subsequent turns: acts based on current belief (I_hat) about Audience's evaluation

2. **Observes** (After Audience responds):
   - Receives the Audience's response (dialogue + body language)
   - Uses an LLM to extract a numeric measurement (0-1) estimating the Audience's internal evaluation
   - This measurement serves as an observation for the particle filter

3. **Updates Belief** (Particle Filter):
   - **Predict step**: Diffuses particles with Gaussian process noise
   - **Update step**: Weights particles by likelihood of the observed measurement
   - **Resample** (if needed): When effective sample size drops below threshold
   - Computes posterior mean I_hat (belief about Audience's evaluation)
   - Records prediction error: PE = |previous I_hat - current I_hat|

4. **Learns** (Reflection):
   - Generates a brief reflection on what to change next turn
   - Uses current I_hat to inform the reflection

### The Audience (Interviewer)

**Goal**: Evaluate the Actor's competence (no target value; evaluation is the output)

**What the Audience Observes**:

1. **Cultural Norms**:
   - Explicit knowledge of all 18 cultural norms (e.g., "Direct, literal language", "Integrity over politeness")
   - **Full 2A25 world description**: Complete world-building context including:
     - Cadens as "the dominant cognitive-cultural order"
     - Riffers as "a social minority group with unique cultural knowledge and individual traits that differ from the Caden-majority"
     - Explanation that "Riffers are stigmatized and need to adopt the norms and behaviors of Cadens to be successful in social interactions"
     - Instructions that norms must be followed strictly and violations lead to negative evaluation

2. **World Context** (if enabled):
   - Interview setting: "an interview room in a corporate office"
   - Role mention: "the role of product manager"
   - Institutional context: "This is a Caden-majority institutional setting"
   - Note: Detailed role requirements (responsibilities, evaluation criteria) are defined but not explicitly included in prompts

3. **Personality Traits**:
   - Their own behavioral profile (e.g., detail-focused, avoids eye contact, not social)
   - Initialized separately via a dedicated prompt before conversation starts
   - Traits shape evaluation style and communication throughout the interaction

4. **Goal Information**:
   - Goal name: "evaluate_competence"
   - Goal description: Instructions on evaluating the Actor's competence

5. **Actor's Utterances**:
   - The Actor's dialogue (text)
   - The Actor's body language descriptions

6. **Conversation History**:
   - Recent conversation history (last K turns, default K=3)
   - Includes both Actor's and Audience's previous utterances

7. **Internal Evaluation State**:
   - Their own I_t value (the evaluation score they assign)
   - Used to generate responses that match their true evaluation

**Key Asymmetry**:
- The Audience has **explicit knowledge** of cultural norms, while the Actor does not
- This creates a learning challenge: the Actor must infer appropriate behavior from feedback
- The Audience uses norms to evaluate whether the Actor's behavior is appropriate

**What the Audience Does**:

1. **Evaluates** (Hidden State I_t):
   - Forms an internal evaluation I_t ∈ [0,1] of the Actor's competence
   - This is the "ground truth" that the Actor is trying to infer
   - Based on Actor's utterance, body language, and evaluation criteria

2. **Responds**:
   - Generates an initial response that reflects I_t
   - If cultural norms or personality traits are enabled:
     - Self-reflects and improves the response to align with norms/traits
     - Produces a final response that matches I_t while conforming to behavioral constraints
   - Response includes both dialogue and body language

3. **Maintains Consistency**:
   - Responses must match the internal evaluation I_t
   - Cultural norms constrain how feedback is expressed
   - Personality traits shape communication style

---

## Information Asymmetry Summary

| Observation | Actor | Audience |
|------------|-------|----------|
| **Cultural Norms** | ❌ No explicit knowledge | ✅ Full knowledge (18 norms) |
| **World Context** | ✅ Yes (if enabled) | ✅ Yes (if enabled) |
| **Personality Traits** | ✅ Yes (low profile: 0-1/3) | ✅ Yes (high profile: 2-3/3) |
| **Interview Setting** | ✅ Yes (if enabled) | ✅ Yes (if enabled) |
| **Role Requirements** | ⚠️ Mentioned only | ⚠️ Mentioned only |
| **Conversation History** | ✅ Yes (last K turns) | ✅ Yes (last K turns) |
| **Partner's Responses** | ✅ Yes | ✅ Yes |
| **Own Belief History** | ✅ Yes (I_hat values) | ❌ N/A |
| **Own I_t Evaluation** | ❌ No (hidden) | ✅ Yes (generates it) |
| **2A25 World Details** | ⚠️ Minimal only | ✅ Full description (via norms) |
| **Reflections** | ⚠️ Generated but not in prompts | ❌ N/A |

**Key Design Features**:
- The Actor must **learn** appropriate behavior through feedback, as they lack explicit knowledge of the cultural norms that govern the Audience's evaluation
- The Audience has **full world-building context** (Cadens vs. Riffers) that the Actor does not see
- Both agents observe the interview setting and role mention, but detailed role requirements are not explicitly included in prompts
- Personality traits are initialized separately before conversation starts, creating stable behavioral profiles
- This creates a realistic social learning scenario with information asymmetry

---

## Defining Features of the Study

### 1. **Particle Filter Belief Tracking**

The Actor maintains a probabilistic belief about the Audience's evaluation using a particle filter:
- **200 particles** distributed over [0,1]
- **Process noise** (σ = 0.03): Models uncertainty about how the Audience's evaluation might change
- **Observation noise** (σ = 0.06): Models uncertainty in extracting measurements from responses
- **Systematic resampling**: Prevents particle degeneracy when effective sample size < 50% of particles

**Key Metrics**:
- **I_hat**: Posterior mean belief (Actor's estimate of Audience's evaluation)
- **I_t**: True Audience evaluation (ground truth)
- **PE**: Prediction error = |previous I_hat - current I_hat|

### 2. **Prediction Error as a Learning Signal**

Prediction errors drive adaptive behavior:
- Large PE → Actor's belief changed significantly → potential learning opportunity
- Small PE → Actor's belief was accurate → confirmation
- PE is computed each turn and stored in memory
- Learning gain = |ΔI_hat| / |PE| (measures how much belief changed relative to error magnitude)

### 3. **Cultural Norms and Personality Traits**

**Cultural Norms** (for Audience only):
- 18 predefined norms (e.g., "Direct, literal language", "Integrity over politeness")
- **Only the Audience has explicit knowledge** of these norms in their prompts
- Norms govern how the Audience evaluates and responds
- Violations of norms lead to negative evaluations
- Simulates a "Caden-majority" cultural context where norms are strictly enforced
- The Actor must **infer** appropriate behavior from feedback, creating a learning challenge

**Personality Traits** (for both agents):
- 11 traits (e.g., "Detail-focused", "Avoids eye contact", "Not social")
- **Audience**: traits set to maximum (3/3) - strong behavioral profile
- **Actor**: traits set to minimum (0/3) - contrasting profile
- Traits are **initialized separately** via `initialize_personality_traits()` before conversation starts
- This initialization uses a dedicated prompt that sets the behavioral profile
- Traits shape communication style and interpretation throughout the interaction
- Both agents observe their own traits, creating distinct behavioral profiles

**World Context** (for both agents, if enabled):
- Both agents observe the interview setting and 2A25 world information
- Includes: interview room description, role requirements, institutional context
- The Actor knows they're in a Caden-majority setting but doesn't know the specific norms
- Creates shared situational awareness while maintaining information asymmetry about norms

### 4. **Interview Context**

When enabled, the study uses a structured interview scenario:
- **Role**: Product Manager position
- **Evaluation Criteria**: Technical understanding, prioritization, communication, data-driven decision making
- **Setting**: Corporate interview room in the 2A25 fictional world
- **Stakes**: Actor must demonstrate competence; Audience evaluates for hiring decision
- **Shared Knowledge**: Both agents observe the interview context, role requirements, and institutional setting
- **Asymmetric Knowledge**: Only Audience knows the cultural norms that govern appropriate behavior

### 5. **Multi-Modal Communication**

Each utterance includes:
- **Dialogue**: Verbal content
- **Body Language**: Non-verbal cues (e.g., "maintains steady eye contact", "avoids eye contact repeatedly")
- Both modalities inform evaluation and belief updates

### 6. **Self-Reflection Mechanisms**

Both agents can self-reflect to improve alignment:
- **Audience**: Reflects on responses to ensure alignment with cultural norms and personality traits
- **Actor**: Reflects on responses to ensure alignment with personality traits
- This creates more consistent and contextually appropriate behavior

### 7. **Comprehensive Logging**

Each turn logs:
- Speaker and listener utterances (dialogue + body language)
- True Audience evaluation (I_t)
- Actor's belief (I_hat)
- Prediction error (PE)
- Effective sample size (ESS)
- Personality and context checks
- Reflection text

Outputs include:
- JSON log with full turn-by-turn data
- Three plots: PE over time, I_t vs I_hat, Learning gain

---

## Interaction Example: A Single Turn

**Turn t = 3**:

1. **Actor Acts**:
   - Current belief: I_hat = 0.63 (from previous turn)
   - Generates: "I effectively communicated product strategies across our organization by showcasing data-backed insights in presentations."
   - Body: "Maintain steady eye contact and nod slightly for emphasis."

2. **Audience Evaluates**:
   - Observes Actor's utterance
   - **Uses cultural norms** to assess whether Actor's behavior is appropriate
   - **Uses personality traits** to shape evaluation style
   - Forms hidden evaluation: I_t = 0.9 (high competence)
   - Generates initial response reflecting this score

3. **Audience Self-Reflects** (if norms/traits enabled):
   - **Checks alignment with cultural norms** (e.g., "Direct, literal language", "Integrity over politeness")
   - **Adjusts response to match personality traits** (e.g., detail-focused, avoids eye contact)
   - Final response: "The interviewee showed notable competence in data-driven decision making, but requires significant improvement in technical understanding and stakeholder communication."
   - Body: "Avoided eye contact repeatedly." (matches Audience's personality trait)

4. **Actor Observes**:
   - Receives Audience's response
   - LLM extracts measurement: 0.6 (interprets the response as moderately positive)

5. **Actor Updates Belief**:
   - Particle filter predict step (diffuses particles)
   - Particle filter update step (weights by measurement likelihood)
   - Computes new I_hat = 0.67
   - Computes PE = |0.63 - 0.67| = 0.04

6. **Actor Learns**:
   - Reflects: "To enhance my perceived competence, I will focus on demonstrating clear examples of data-driven decision-making and effective stakeholder communication."
   - **Note**: Actor does not explicitly know the cultural norms but may infer appropriate behavior patterns from the Audience's feedback style

---

## Key Classes and Functions

### Core Data Structures

**`Goal`**: Defines agent objectives
- `name`: Goal identifier (e.g., "competence")
- `description`: Detailed goal description
- `role`: Context-specific role information
- `ideal`: Target value (1.0 for Actor, None for Audience)

**`Utterance`**: Represents a single communication
- `turn`: Turn number
- `speaker`: Agent name
- `text`: Dialogue content
- `body`: Body language description

**`AgentMemory`**: Stores agent's cognitive state
- `conversation`: History of utterances
- `pe_history`: Prediction error records
- `reflections`: Learning reflections
- `pf_particles`, `pf_weights`: Particle filter state
- `pf_history`: Belief tracking history

### Key Classes

**`Agent`**: Main agent class
- **`act()`**: Initial utterance generation
- **`act_based_on_belief()`**: Utterance generation conditioned on belief
- **`audience_evaluate_and_respond()`**: Audience evaluation and response
- **`actor_update_particles()`**: Particle filter belief update
- **`learning()`**: Generate reflection for next turn

**`ParticleFilter`**: Probabilistic belief tracking
- **`initialize()`**: Set up initial particle distribution
- **`predict()`**: Apply process noise (belief diffusion)
- **`update()`**: Weight particles by observation likelihood
- **`_systematic_resample()`**: Resample to prevent degeneracy

**`ConversationStudy`**: Orchestrates the interaction
- **`run()`**: Executes the turn-by-turn conversation loop
- **`save_json()`**: Persists logs to file

### Key Functions

**`plot_learning_dynamics()`**: Generates three diagnostic plots
1. Prediction Error over turns
2. True evaluation (I_t) vs. estimated belief (I_hat)
3. Learning gain (belief change per unit of error)

**`generate_parametric_traits()`**: Assigns personality traits
- Audience: Maximum trait values (strong profile)
- Actor: Minimum trait values (contrasting profile)

---

## Experimental Parameters

### Configurable Settings

- **`--turns`**: Number of conversation turns (default: 2)
- **`--model`**: LLM model (default: "gpt-4o-mini")
- **`--temperature`**: Sampling temperature (default: 0.2)
- **`--window`**: Recent K turns for context (default: 3)
- **`--no_audience_norms`**: Disable cultural norms
- **`--no_traits`**: Disable personality traits
- **`--no_context`**: Disable interview context
- **`--seed`**: Random seed for reproducibility

### Particle Filter Parameters

- **`num_particles`**: 200 (fixed)
- **`process_sigma`**: 0.03 (belief diffusion rate)
- **`obs_sigma`**: 0.06 (measurement uncertainty)

---

## Outputs and Analysis

### Generated Files

1. **`pe_conversation_log.json`**: Complete turn-by-turn log
   - All utterances, evaluations, beliefs, PEs, reflections
   - Personality and context checks

2. **`pe.png`**: Prediction error over time
   - Shows how PE evolves across turns

3. **`delta_I.png`**: True vs. estimated evaluation
   - I_t (ground truth) vs. I_hat (Actor's belief)
   - Measures tracking accuracy

4. **`learning_gain.png`**: Learning efficiency
   - Ratio of belief change to prediction error
   - Higher values indicate more efficient learning

### Analysis Metrics

- **Tracking Accuracy**: Correlation between I_t and I_hat
- **PE Magnitude**: Average prediction error size
- **Learning Rate**: Rate of I_hat convergence to I_t
- **Response Alignment**: Consistency between I_t and Audience responses

---

## Design Rationale

### Why Particle Filters?

Particle filters provide:
- **Uncertainty quantification**: Maintains a distribution, not just a point estimate
- **Non-linear dynamics**: Can handle complex belief updates
- **Robustness**: Resampling prevents particle collapse
- **Interpretability**: ESS provides diagnostic information

### Why Prediction Error?

Prediction errors are:
- **Universal learning signal**: Used across cognitive systems
- **Quantifiable**: Easy to measure and track
- **Actionable**: Large errors trigger behavioral adjustments
- **Theoretically grounded**: Links to reinforcement learning and Bayesian inference

### Why Cultural Norms and Traits?

These features:
- **Add realism**: Simulate real-world social constraints
- **Create information asymmetry**: Only Audience knows norms explicitly; Actor must infer from feedback
- **Enable hypothesis testing**: Can study norm violation effects and learning dynamics
- **Increase complexity**: More realistic impression management scenarios
- **Model social learning**: Actor must learn appropriate behavior through interaction, not explicit instruction
- **Create behavioral contrast**: Different trait profiles (Actor: low traits, Audience: high traits) create distinct communication styles

---

## Future Extensions

Potential enhancements:
- Multi-dimensional goals (not just competence)
- Multiple audience members
- Dynamic norm learning
- Explicit deception detection
- Long-term memory integration
- Cross-cultural comparisons

---

## References and Notes

- The study uses OpenAI's Responses API for LLM calls
- Cultural norms simulate a fictional "2A25" world with "Cadens" and "Riffers"
- All interactions are logged for reproducibility
- The design supports both interview and generic conversation contexts
