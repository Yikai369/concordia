# ðŸ§ª Experiment Specification: Neurotype-Based Norm Adherence in Concordia

## Overview

This experiment simulates interactions between two agents â€” a **candidate** and an **interviewer** â€” within a shared environment governed by **Riffer social norms**.

The interaction is framed as a job interview for a customer service role, where both agents:

* Understand the job requirements
* Operate under Riffer communication norms
* Engage in a semi-structured conversational interview

The goal is to study:

* How agents with different **neurotypes** (Riffer vs Caden) adhere to the same norms
* How this affects:
  * **Norm adherence**
  * **Satisfaction**
  * **Surprise**

---

## Core Hypothesis

Even when both agents:

* Operate under the **same norms**
* Have **explicit knowledge** of those norms

â†’ Differences in **neurotype (personality + cognition)** will result in:

* Different levels of **norm adherence**
* Different **interaction outcomes** (measured by satisfaction and surprise)

---

## Experimental Variables

### Independent Variables

* **Candidate neurotype**

  * Riffer
  * Caden

* **Interviewer neurotype**

  * Riffer
  * Caden

---

### Controlled Variables

* Shared **environmental norms** (Riffer norms)
* Same **interaction structure**
* Same **questions/prompts**
* Same **memory generation process**

---

### Dependent Variables

* **Norm adherence (per agent)**
* **Satisfaction (per agent)**
* **Surprise (per agent)**

---

## System Architecture

```
Agents:
    - Candidate
    - Interviewer

Game Master:
    - Defines norms
    - Orchestrates interaction
    - Evaluates metrics

Initializer:
    - Generates personality + memory

Simulation:
    - Runs interaction loop
```

---

## Component Specifications

---

### 1. Agent Configuration (`agent_config`)

Each agent is defined with:

* Role: `"candidate"` or `"interviewer"`
* Neurotype: `"Riffer"` or `"Caden"`
* Traits (natural language) or None
* Formative memory prompts or None
* Context (must include norm awareness)
* Behavioral instructions (see Section 2)

---

### Example: Candidate (Riffer)

```python
candidate_config = {
    "name": "Candidate",
    "role": "candidate",
    "neurotype": "Riffer",
    "traits": """
    Prefers direct and explicit communication, avoids implicit cues,
    values clarity and honesty, may struggle with rapid conversational pacing,
    tends toward literal interpretation.
    """,
    "formative_memory_prompts": [...],
    "context": """
    You have learned and understand Riffer communication norms,
    including directness, minimal reliance on non-verbal cues,
    and valuing honesty over politeness.
    """
}
```

---

### Example: Interviewer (Caden)

```python
interviewer_config = {
    "name": "Interviewer",
    "role": "interviewer",
    "neurotype": "Caden",
    "traits": """
    Uses indirect phrasing, relies on tone and social smoothing,
    adapts conversationally, tends to soften statements,
    comfortable with implicit communication.
    """,
    "formative_memory_prompts": [...],
    "context": """
    You have learned and understand Riffer communication norms,
    including directness, minimal reliance on non-verbal cues,
    and valuing honesty over politeness.
    """
}
```

### Shared Knowledge: Job Description (BOTH AGENTS)

Both agents must have access to the same job description:

```python
JOB_DESCRIPTION = """
Customer Service Representative Role:

Responsibilities:
- Communicate clearly and directly with customers
- Resolve issues efficiently and accurately
- Ask clarifying questions when needed
- Maintain professionalism and composure

Evaluation Criteria:
- Clarity of communication
- Problem-solving ability
- Responsiveness and adaptability
- Ability to follow communication norms
"""
```
This must be included in:

* agent context

### Candidate Context (MUST include awareness of evaluation + norms)
```python
context = """
You are applying for a customer service role.

You understand the job requirements and will be evaluated on:
- communication clarity
- problem-solving ability
- professionalism

The interviewer is part of the majority group in this environment,
which follows Riffer communication norms:
- direct
- explicit
- minimal reliance on non-verbal cues

You are aware that:
- you will be judged based on these norms
- deviations may negatively affect evaluation

Your goal is to perform well in the interview while managing your natural tendencies.
"""
```

### Interviewer Context
```python
context = """
You are conducting an interview for a customer service role.

You evaluate candidates based on:
- clarity
- directness
- problem-solving ability
- adherence to communication norms

You follow Riffer communication norms and expect candidates to do the same.

You will ask questions to assess competence while maintaining a natural conversational flow.
"""
```
---
## 2. Behavioral Instruction Layer (Agent Policy)
### Purpose:
Defines real-time decision-making behavior for agents during interaction.

This layer ensures agents:

* Attempt to follow norms
* Do not behave perfectly
* Exhibit internal conflict between knowledge and natural tendencies
Design Principle:
knowledge of norms
+ natural tendencies
+ real-time decision pressure
â†’ final response

### Integration Point

Behavioral instructions must be included in the agent action prompt:
```python
def build_agent_prompt(agent, observation):
    return f"""
    {agent.identity}

    Relevant memories:
    {retrieve_memories(agent, observation)}

    Situation:
    {observation}

    Instructions:
    {agent.behavioral_instructions}

    Respond as the agent.
    """
```

Candidate (Minority / Masking)
```python
behavioral_instructions = """ When responding:
- You are being evaluated for a job
- You are trying to follow Riffer communication norms:
  - direct
  - explicit
  - clear
- You want to appear competent and aligned with expectations
- You actively monitor:
  - whether your response is clear
  - whether it follows norms

However:
- your natural tendencies may still influence your response
- under pressure, you may revert to default communication patterns

Balance:
- performing well in the interview
- adhering to norms
- your natural tendencies """
```

Interviewer (Majority / Norm Enforcer)
```python
behavioral_instructions = """
When responding:
- You are assessing the candidate for a customer service role
- You ask questions to evaluate competence

You should:
- draw from relevant interview questions
- adapt based on previous responses
- maintain a natural conversational flow

You evaluate based on:
- clarity
- directness
- problem-solving ability
- adherence to norms

If the candidate deviates:
- you may notice
- it may influence your evaluation """

```
---

## 3. Formative Memory Initialization

Use `formative_memories_initializer`.

### Requirements

* Generate **20â€“50 memories per agent**
* Memories must:

  * Reflect personality traits
  * Include social experiences
  * Reinforce understanding of Riffer norms
  * Job-related experiences
  * Communication experiences

---

### Memory Prompt Guidelines

Use diverse, specific prompts such as:

```python
formative_memory_prompts = [
    "Describe a childhood experience that shaped how you communicate.",
    "Describe a time you misunderstood someone in a social setting.",
    "Describe a situation where direct communication helped or hurt you.",
    "Describe how you learned social norms in your environment.",
    "Describe a moment where you felt misunderstood.",
    "Describe a time you helped someone solve a problem.",
    "Describe a situation where you had to communicate clearly under pressure.",
    "Describe a time you misunderstood instructions and what happened.",
    "Describe how you learned what is expected in professional communication.",
    "Describe a situation where you struggled to meet expectations.",
]
```

---

## 4. Game Master (Custom Implementation)

Extend `psychology_experiment.GameMaster`.

---

### A. Shared Norms Definition

```python
GM_CONTEXT = """
Riffer social norms:

- Direct and explicit communication is preferred
- Minimal reliance on non-verbal cues
- Honesty is valued over politeness
- Pauses in conversation are acceptable
- Overly expressive tone may feel overwhelming
"""
```

---

### B. Interaction Loop

The interview must follow a **structured but natural conversational flow**:

---

### Turn 0: Greeting Phase

```
Interviewer â†’ Candidate
```

* The interviewer **initiates the conversation**
* Provides a greeting and sets the tone
* May briefly introduce the interview context

---

### Subsequent Turns (Core Loop)

For each turn:

```
Candidate â†’ Interviewer â†’ Candidate â†’ Interviewer â†’ ...
```

---

### Candidate Behavior

* Responds to interviewer
* May:

  * Draw from **past experiences (memories)**
  * Reference relevant examples
  * Elaborate on previous answers
* Should:

  * Maintain conversational continuity
  * Stay aligned with job context

---

### Interviewer Behavior

* Responds to candidate
* May:

  * Use the question bank as inspiration to evaluate candidate's competency
  * Ask follow-up questions based on candidate responses
  * Probe deeper into previous answers
  * Adapt questions to the current conversation

The question bank must **NOT** be used as a rigid script.

### Interaction Loop
Turn Structure

Turn 0:
    Interviewer greets candidate

Turn t â‰¥ 1:
    Candidate responds
    Interviewer responds

Pseudocode
```python
# Turn 0: Greeting
interviewer_message = interviewer.greet()
for t in range(1, T):
  # Candidate responds (can use memories)
  candidate_response = candidate.act( observation=interviewer_message, allow_memory_retrieval=True)
  # Interviewer responds (guided by question bank)
  interviewer_message = interviewer.act( observation=candidate_response, question_bank=QUESTION_BANK, conversational=True)
```

### C. Adherence Friction (Critical)

Agents may NOT perfectly follow norms.

Implement:

* Caden agent:

  * May revert to indirect phrasing
  * May use unnecessary social smoothing

* Riffer agent:

  * May struggle with pacing
  * May be overly literal

Methods:

* Prompt-based internal conflict
* Optional probabilistic adherence

---

## 5. Metrics Module

All evaluations should be performed via LLM prompts.

---

### A. Norm Adherence

```python
evaluate_adherence(response, norms) -> score (1â€“10)
```

Criteria:

* Directness
* Explicitness
* Avoidance of implicit cues

---

### B. Satisfaction

```python
evaluate_satisfaction(agent, interaction) -> score
```

Criteria:

* Comfort
* Clarity
* Perceived effectiveness

---

### C. Surprise

```python
evaluate_surprise(agent, interaction) -> score
```

Criteria:

* Deviation from expectations
* Unpredictability

---

## 6. Logging System

Log per interaction round:

```python
{
  "turn": t,
  "candidate_neurotype": ...,
  "interviewer_neurotype": ...,

  "interviewer_message": ...,
  "candidate_response": ...,

  "candidate_adherence": ...,
  "interviewer_adherence": ...,

  "candidate_competence": ...
}
```

---

### Storage Format

* JSON (preferred for flexibility)
* CSV (for analysis)

---

## 6. Experiment Command Line Args

---

### Conditions to Specify

```
1. Candidate (Riffer) â†” Interviewer (Riffer)
2. Candidate (Caden) â†” Interviewer (Caden)
3. Candidate (Riffer) â†” Interviewer (Caden)
4. Candidate (Caden) â†” Interviewer (Riffer)
```

---

## Key Implementation Requirements

---

### 1. Norm Awareness (Mandatory)

Both agents MUST:

* Explicitly know Riffer norms
* Have this encoded in memory/context

---

### 2. Imperfect Adherence

Agents must:

* Understand norms
* Fail to consistently follow them when in contradiction with personality traits

This is essential to the experiment.

---

### 3. Role Separation

* Candidate and interviewer are both **agents**
* Game Master is **not** an agent
* Game Master handles:

  * orchestration
  * evaluation
  * logging

---

### 4. Memory Depth

Each agent must have:

* **20â€“50 formative memories**
* Rich, diverse, and personality-consistent experiences

---

## Expected Outputs

---

### 1. Adherence Analysis

* Compare norm adherence across neurotypes

---

### 2. Satisfaction Patterns

* Prompt for agent satisfaction each turn in interaction

---

### 3. Surprise Patterns

* Prompt for agent surprise each turn in interaction

---

## Conceptual Model

```
Norms (external)
    +
Neurotype (internal)
    â†’
Behavior (interaction)
    â†’
Outcomes (adherence, satisfaction, surprise)
```

---

## Deliverable

A reproducible pipeline that:

1. Initializes agents with personality and 20â€“50 memories
2. Runs structured interactions under shared Riffer norms
3. Measures adherence, satisfaction, and surprise
4. Logs results across all conditions

---

Recreate a version of the impression_management project, except using more components from concordia, as specified in im_gm.md. The main.py loop from impression_management project should be fully replaced by the game master, as well as some formative memory initializers for the agents. Preserve constants in constants.py for world building, norms, interview context/customer service role, names, and particle filter parameters. This new project should refactor necessary components from concordia/components/agent/impression_management_pe.py, concordia/prefabs/entity/impression_management_actor.py, and concordia/prefabs/entity/impression_management_audience.py, preserving relevant aspects and using them as inspiration to meet the specifications in im_gm.md.

# 🧠 Formative Memory Generator Prompt Template

## Purpose

Generate a set of **20–50 formative memories** that collectively form a realistic, behaviorally rich backstory for an agent.

---

## Input Variables

```python
NORMS = """{norms}"""

TRAITS = """{trait_paragraph}"""

SEED_PROMPTS = {prompt_list}
```

---

## Full Prompt Template

```text
You are generating formative life memories for an individual.

----------------------------------------
WORLD CONTEXT (BACKGROUND CULTURE)
----------------------------------------
This individual lives in a society where the following communication norms are typical:

{NORMS}

These norms shape everyday interactions in school, work, and relationships.

IMPORTANT:
- Do NOT explicitly list or restate these norms in the memories
- Instead, let them appear naturally through situations, expectations, and reactions

----------------------------------------
INDIVIDUAL BACKGROUND (LATENT TRAITS)
----------------------------------------
This individual has the following underlying tendencies:

{TRAITS}

IMPORTANT:
- Do NOT explicitly mention or restate these traits
- Do NOT describe the person using labels
- Instead, let these traits influence:
  - what situations stand out
  - how events are interpreted
  - how the individual reflects on experiences

----------------------------------------
MEMORY GENERATION INSTRUCTIONS
----------------------------------------
Generate a diverse set of formative memories across the individual's life.

Each memory should:
- Be specific and grounded (not generic summaries)
- Include a clear situation, action, and outcome
- Include internal thoughts or reactions when appropriate
- Include a brief reflection on what the individual took away

Memories should span a variety of contexts, including:
- childhood and upbringing
- school and structured environments
- friendships and relationships
- group dynamics and belonging
- conflict and misunderstanding
- work or responsibility
- stress or uncertainty
- identity and personal values

Only some memories should directly involve communication or social norms.
Others should involve general life experiences.

----------------------------------------
STYLE CONSTRAINTS
----------------------------------------
- Avoid repetitive phrasing or patterns
- Avoid overly formal or robotic language
- Do NOT explain behavior using abstract traits
- Show behavior through concrete experiences
- Reflections should feel natural, not analytical essays

----------------------------------------
SEED PROMPTS
----------------------------------------
Use the following prompts as inspiration for individual memories:

{SEED_PROMPTS}

Each prompt should loosely guide one memory, but:
- You may reinterpret or expand prompts creatively
- You may add additional realistic memories beyond these prompts

```
