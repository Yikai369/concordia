# Impression Standard Game: Component Summary

This document summarizes all relevant components used in the **Impression Management PE (Prediction Error) Standard** game—a two-agent interview simulation that uses the Concordia framework and a particle filter for belief tracking.

---

## 1. Overview

- **Purpose**: Simulate a dyadic interview where an **Actor** (interviewee) tries to be perceived as competent by an **Audience** (interviewer). The Actor maintains a particle-filter belief over the Audience’s hidden evaluation state and adapts behavior based on prediction errors.
- **Loop**: Standard Concordia simulation loop (`sim.play()`). Each logical “turn” = Actor acts, then Audience observes (evaluates) and acts.
- **Project**: `projects/impression_management_standard/` (entry: `main.py`).

---

## 2. Project Structure (`projects/impression_management_standard/`)

| File / Folder | Role |
|---------------|------|
| `main.py` | Entry point: parse config, setup model/embedder/memory, create simulation config, run `sim.play()`, extract and save results. |
| `config.py` | CLI via `argparse`; builds `ConversationConfig` (see `models.py`). |
| `models.py` | `ConversationConfig` (all run settings), `TurnLog` (per-turn output). |
| `simulation_config.py` | Builds Concordia `Config`: prefabs, instances (actor, audience, game master), default premise, max steps. |
| `constants.py` | Cultural norms, personality traits, interview role presets, question/experience banks, defaults (particles, sigma, names, etc.). |
| `utils.py` | Output directory creation, trait score generation, optional `load_traits_from_spreadsheet()`. |
| `setup.py` | `setup_language_model()` (OpenAI or Ollama), `setup_embedder_and_memory()` (sentence-transformers + `AssociativeMemoryBank`). |
| `simple_audience_prefab.py` | **Audience** entity prefab for the standard loop (evaluate on observe, act by returning stored response). |
| `audience_act_component.py` | `SimpleAudienceActComponent`: returns stored evaluation response from IMPE memory. |
| `data_extraction.py` | `extract_turn_data_from_entities()`: pulls conversation, evaluations, PF history, reflections, PE from entities into `TurnLog` list. |
| `results.py` | Save turn logs (JSON), optional question checks, component logs, info-flow/simplified logs. |
| `docs/` | Extra docs (e.g. `component_architecture.md`, CLI, improvements). |

---

## 3. Cognitive Components (one-sentence descriptions)

All agent-side components that contribute to perception, memory, reasoning, or action in the impression standard game, with a single-sentence description each.

| Component | One-sentence description |
|-----------|--------------------------|
| **Instructions** | Provides role-playing and experimental context so the agent knows it is in an interview. |
| **SelfPerception** | Answers “What kind of person am I?” from recent memories to shape self-concept. |
| **SituationPerception** | Answers “What kind of situation am I in right now?” from recent observations. |
| **PersonBySituation** | Answers “What would a person like me do in this situation?” by combining self- and situation perception. |
| **AssociativeMemory** | Stores and retrieves general observations (embedding-based) for context. |
| **ObservationToMemory** | Writes incoming observations into associative memory during observe. |
| **IMPEMemoryComponent** | Holds conversation history, evaluation records (I_t), particle-filter state, reflections, and optional conversation summary. |
| **CulturalNormsComponent** | Supplies the list of cultural norms the agent should follow (e.g. for evaluation or behavior). |
| **PersonalityTraitsComponent** | Supplies personality traits (scores or an LLM-generated paragraph) that shape how the agent behaves or is evaluated. |
| **WorldContextComponent** | Supplies 2A25 world-building (Cadens/Riffers) and interview-setting context. |
| **IMPEAudienceEvaluationComponent** | On observe: infers the true evaluation I_t of the actor and generates the audience’s feedback utterance. |
| **IMPEActorParticleFilterComponent** | Maintains a particle-filter belief (I_hat) over the audience’s evaluation and updates it with each observation. |
| **IMPEReflectionComponent** | Produces a short reflection on how to improve toward the goal given the current I_hat. |
| **IMPEActComponent** | Generates the actor’s utterance from goal, I_hat, history, reflections, norms, traits, and world context. |
| **IMPESelfAssessmentComponent** | Scores the planned response for consistency with traits, norms, and goal, and optionally revises it if below threshold. |
| **SimpleAudienceActComponent** | Returns the audience’s stored evaluation response (the utterance produced by IMPEAudienceEvaluationComponent on observe). |

---

## 4. Concordia Prefabs and Game Master

### 4.1 Entity Prefabs

| Prefab | Module | Description |
|--------|--------|-------------|
| **Impression Management Actor** | `concordia.prefabs.entity.impression_management_actor` | Interviewee: particle filter (I_hat), reflection, IMPE act, optional instructions/self-perception/situation/per-person-by-situation, world context, norms/traits, self-assessment. |
| **Simple Audience** | `projects.impression_management_standard.simple_audience_prefab` | Interviewer: same optional context components; **evaluates on observe** (IMPEAudienceEvaluationComponent), **acts** by returning stored response via `SimpleAudienceActComponent`; optional self-assessment. |
| **Impression Management Audience** (alternative) | `concordia.prefabs.entity.impression_management_audience` | Full audience prefab (e.g. for non–standard-loop use); uses `Constant` act (no speaking) and IMPE audience evaluation. |

### 4.2 Game Master

| Prefab | Module | Description |
|--------|--------|-------------|
| **IMPE Game Master** | `concordia.prefabs.game_master.impression_management_pe` | Fixed order: **Actor** then **Audience** each step; standard GM components (instructions, player characters, observations, display events, make observation, send events, next actor, next action spec, repetitive-conversations end, next game master, event resolution). Optional `NeverTerminate` when `can_terminate_simulation=False`. |

---

## 5. IMPE Components (`concordia/components/agent/impression_management_pe.py`)

### 5.1 Data Classes and Helpers

| Name | Description |
|------|-------------|
| `Goal` | From `pe_conversation`: name, description, role, ideal (e.g. competence 0–1). |
| `Utterance`, `PERecord`, `ReflectionRecord` | From `pe_conversation`. |
| `EvaluationRecord` | turn, I_t (true evaluation), utterance. |
| `ObservationRecord` | turn, observed_from, text, body. |
| `ActionRecord` | turn, text, body. |
| `CulturalNorm` | name, description. |
| `PersonalityTrait` | name, assertion. |
| `ParticleFilter` | 1D particle filter on [0,1]: init, predict (process noise), update (observation likelihood, resample). |

### 5.2 Component Keys (constants)

- `IMPE_Memory`, `IMPE_AudienceEvaluation`, `IMPE_ActorParticleFilter`, `IMPE_Reflection`, `IMPE_Act`, `CulturalNorms`, `PersonalityTraits`, `WorldContext`.

### 5.3 Core Components

| Component | Role |
|-----------|------|
| **IMPEMemoryComponent** | Extends PE memory: conversation history, evaluation history, PF state (particles/weights), PF history, observation/action history, optional conversation summary cache (memory check). |
| **CulturalNormsComponent** | Exposes cultural norms; optional one-time `initialize_norms(model, entity_name)`. |
| **PersonalityTraitsComponent** | Exposes traits (scores or LLM-generated paragraph). |
| **WorldContextComponent** | 2A25 world-building (Cadens/Riffers) and/or interview context; `use_full_2a25` toggles full vs minimal. |
| **IMPEAudienceEvaluationComponent** | On observe: parse actor DIALOGUE/BODY, compute I_t (LLM), generate feedback utterance, store in IMPE memory. Optional option-space and memory-check. |
| **IMPEActorParticleFilterComponent** | Maintains particle filter over audience evaluation; on pre_act: predict, then update using latest observation (I_t); exposes I_hat and stats. |
| **IMPEReflectionComponent** | Pre_act: generate reflection on how to improve toward goal given current I_hat. |
| **IMPEActComponent** | Act: generate actor utterance from goal, I_hat, history, reflections, norms, traits, world context. Optional option-space and memory-check. |
| **IMPESelfAssessmentComponent** | Wraps a base act component: assess consistency with traits/norms/goals; optionally revise if below `consistency_threshold`; `disable_revision` to only log. |

### 5.4 Optional Wrapper: Self-Assessment

**IMPESelfAssessmentComponent** is an optional wrapper used by both Actor and Audience (controlled by `enable_self_assessment` in instance params, default: enabled in the standard project):

- **Actor**: Wraps `IMPEActComponent`. Before returning the generated utterance, the component asks the LLM to score how consistent the response is with the agent’s personality traits, cultural norms (if any), and goal. If the score is below `consistency_threshold` (default 0.7), it can generate a revised response (unless `disable_revision` is True, in which case it only logs the assessment).
- **Audience**: Wraps `SimpleAudienceActComponent`. Same consistency check and optional revision for the audience’s evaluation response.

CLI: `--enable_self_assessment` / `--no_self_assessment`, `--consistency_threshold`, `--disable_revision`.

---

## 6. Shared Agent Components (from Concordia)

Used by both actor and audience prefabs where applicable:

| Component | Module | Role |
|-----------|--------|------|
| **Instructions** | `concordia.components.agent.instructions` | Role-playing instructions. |
| **SelfPerception** | `question_of_recent_memories` | “What kind of person is {name}?” |
| **SituationPerception** | `question_of_recent_memories` | “What kind of situation is {name} in?” |
| **PersonBySituation** | `question_of_recent_memories` | “What would a person like {name} do in this situation?” (depends on Self + Situation). |
| **AssociativeMemory** | `agent_components.memory` | General observations. |
| **ObservationToMemory** | `agent_components.observation` | Write observations to associative memory. |

---

## 7. Execution Flow

1. **main.py**: Parse `ConversationConfig`, validate API key, setup model + embedder + memory, resolve traits (file / constants / none).
2. **simulation_config.create_simulation_config()**: Build prefab dict (including `impression_management_actor__Entity`, `simple_audience__Entity`, `impression_management_pe__GameMaster`), build instance list (actor, audience, game master) with params from config.
3. **Simulation**: `simulation.Simulation(config=sim_config, model=..., embedder=..., ...)` then `sim.play(max_steps=cfg.turns * 2, raw_log=raw_log)`.
4. **Per step**: Game master selects next entity (actor then audience in fixed order). Entity observes (if applicable), then acts. Audience’s observe runs **IMPEAudienceEvaluationComponent** (I_t + response stored); audience’s act returns that stored response (or self-assessed revision).
5. **Post-run**: `data_extraction.extract_turn_data_from_entities()` → list of `TurnLog`. Optional question checks (situation/personality summaries). `results.save_results()` (and optional component logs, info-flow, simplified log).

---

## 8. Information flow across components

How information moves through each entity’s components, by phase. The simulation alternates: one step the **Actor** is chosen (and acts), the next step the **Audience** is chosen (observes the actor’s action, then acts).

### 8.1 Actor (interviewee)

| Phase | What happens | Components involved | Information flow |
|-------|----------------|----------------------|-------------------|
| **Observe** (when GM sends audience utterance) | Actor receives the audience’s last message. | **ObservationToMemory** | Incoming text → AssociativeMemory (and, for IMPE, the actor’s IMPE memory is updated elsewhere when the actor’s own action was processed). |
| **Pre-act** | Context for the next utterance is built. | **Instructions**, **SelfPerception**, **SituationPerception**, **PersonBySituation** (if enabled), **IMPEMemoryComponent**, **IMPEActorParticleFilterComponent**, **IMPEReflectionComponent**, **WorldContextComponent**, **CulturalNormsComponent**, **PersonalityTraitsComponent** | Memory + observations → self/situation/person-by-situation text; IMPE memory + latest I_t → particle filter update (I_hat) and reflection; all → combined context string for the act component. |
| **Act** | Actor produces one reply. | **IMPEActComponent** (optionally wrapped by **IMPESelfAssessmentComponent**) | Context + goal + I_hat → LLM → DIALOGUE + BODY; optionally consistency check and revision. |
| **Post-act / update** | Action is recorded. | **IMPEMemoryComponent** (and any components that persist the action) | Actor’s utterance and body → conversation history and action history; available for PF and reflection on the next turn. |

**Cross-turn flow for the actor:** Each time the actor acts, their utterance is later observed by the audience. When the actor observes the audience’s response, that observation is not processed by the particle filter until the actor’s *next* pre_act (IMPEActorParticleFilterComponent uses the latest evaluation from IMPE memory to update I_hat).

### 8.2 Audience (interviewer)

| Phase | What happens | Components involved | Information flow |
|-------|----------------|----------------------|-------------------|
| **Observe** (when GM sends actor utterance) | Audience receives the actor’s last message. | **ObservationToMemory**, **IMPEAudienceEvaluationComponent** (pre_observe + post_observe) | Incoming DIALOGUE/BODY → AssociativeMemory; same event → audience evaluation: LLM produces I_t and feedback utterance → both stored in **IMPEMemoryComponent**. |
| **Pre-act** | Context for “acting” is built (audience’s act is returning the stored response). | **Instructions**, **SelfPerception**, **SituationPerception**, **PersonBySituation** (if enabled), **IMPEMemoryComponent**, **WorldContextComponent**, **CulturalNormsComponent**, **PersonalityTraitsComponent** | Same pattern as actor for context; no PF or reflection. |
| **Act** | Audience returns the evaluation response already produced in observe. | **SimpleAudienceActComponent** (optionally wrapped by **IMPESelfAssessmentComponent**) | Read latest evaluation utterance from IMPEMemoryComponent → return DIALOGUE + BODY; optionally consistency check and revision. |
| **Post-act / update** | Response is already in IMPE memory from post_observe. | — | No extra write; conversation history already updated when evaluation was stored. |

**Cross-turn flow for the audience:** The audience never “generates” during act in the base design—it only returns the response produced by IMPEAudienceEvaluationComponent during observe. That response is then sent by the game master to the actor as the next observation.

---

## 9. Role of the game master (functional or not?)

**The game master is functional.** This has been checked against the Concordia sequential engine and the IMPE game master prefab.

**How the engine uses the game master each step** (see `concordia/environment/engines/sequential.py` and `concordia/components/game_master/switch_act.py`):

1. **Termination** (start of step): `game_master.act(TERMINATE)` is called. The GM’s context (instructions, player_characters, repetitive_conversations_end, relevant_memories, display_events) is assembled; the GM either uses a **NeverTerminate** component (when `can_terminate_simulation=False`) or the default path asks the LLM “Is the game/simulation finished?”. So the GM actively participates in the stop decision.
2. **Next game master** (only with multiple GMs): With a single IMPE game master, the engine skips this and never calls `act(NEXT_GAME_MASTER)`, so the **NextGameMaster** component is not used for “which GM next.” Termination still uses the GM as above.
3. **Observations**: For each entity, `game_master.act(MAKE_OBSERVATION)` is called with the entity’s name. The GM’s **MakeObservation** component (fed by **DisplayEvents**, **LastNObservations**) produces the observation string; that string is what the entity receives in `observe()`. So the GM is the source of every observation.
4. **Who acts next**: `game_master.act(NEXT_ACTING)` then `game_master.act(NEXT_ACTION_SPEC)`. The GM’s **NextActingInFixedOrder** and **FixedActionSpec** components supply the next entity (actor then audience) and the action spec (e.g. DIALOGUE + BODY). So the GM enforces turn order and action format.
5. **Resolution**: After the entity acts, the engine calls `game_master.observe(putative_event)` then `game_master.act(RESOLVE)`. The GM’s **EventResolution** component runs (for IMPE, a simple text cleanup). The GM then observes the resolved event. So the GM processes every action.

What the game master **does**:

- **Turn order**: Chooses who acts next via **NextActingInFixedOrder** (actor, then audience, repeatedly).
- **Action spec**: Supplies **FixedActionSpec** (DIALOGUE + BODY) so both entities use the same format.
- **Observations**: **DisplayEvents**, **MakeObservation**, and the GM’s observation pipeline produce the string each entity receives in `observe()`.
- **Event resolution**: **EventResolution** runs on each entity action (for IMPE, minimal transformation).
- **Termination**: Either **NeverTerminate** (when disabled) or an LLM decision using the GM’s full context (including “repetitive or long conversations should end”).
- **Own context**: **Instructions**, **PlayerCharacters**, **relevant_memories**, **observation** / **observation_to_memory**, **memory_component** feed into the GM’s decisions and observations.

What the game master **does not** do:

- It does **not** generate interview content; the **Audience** (IMPEAudienceEvaluationComponent) produces I_t and feedback, and the **Actor** (IMPEActComponent) produces replies.
- It does **not** implement IMPE logic (particle filter, reflection, evaluation); that is entirely in the entity components.

**Conclusion**: The game master is the **orchestrator** used by the engine every step for observations, turn order, action spec, resolution, and termination. The **content** of the interview is produced by the Actor and Audience components.

---

## 10. Configuration (ConversationConfig / CLI)

Relevant CLI and config fields (see `config.py` and `models.py`):

- **Run**: `--turns`, `--seed`, `--save_dir`, `--outfile`, `--model`, `--temperature`, `--top_p`, `--window` (recent_k), `--actor_name`, `--audience_name`, `--llm_type`, `--local_model`.
- **Content**: `--no_context`, `--no_audience_norms`, `--no_traits`, `--traits_file`, `--actor_has_norms`, `--interview_role_preset`, `--no_question_bank`, `--no_experience_bank`.
- **World**: `--no_world_building`, `--no_interview_context`, `--no_full_2a25` (use_full_2a25_world).
- **Perception**: `--no_instructions`, `--no_self_perception`, `--enable_situation_perception`, `--no_situation_perception`, `--enable_person_by_situation`, `--no_person_by_situation`.
- **Self-assessment**: `--enable_self_assessment` / `--no_self_assessment`, `--consistency_threshold`, `--disable_revision`.
- **Features**: `--use_trait_paragraph`, `--use_option_space`, `--use_memory_check`, `--enable_question_checks`.
- **Logging**: `--enable_info_flow_logging`, `--enable_simplified_log`, `--simplified_log_format`, `--save_component_logs`, `--pretty_trace`, `--no_plots`.

Constants (particles, sigma, names, roles, norms, traits, banks) live in `constants.py`.

---

## 11. Key File Reference

| Concern | Location |
|---------|----------|
| Entry point | `projects/impression_management_standard/main.py` |
| CLI & config | `config.py`, `models.py` |
| Simulation config | `simulation_config.py` |
| Constants | `constants.py` |
| Actor entity | `concordia/prefabs/entity/impression_management_actor.py` |
| Audience entity (standard) | `projects/impression_management_standard/simple_audience_prefab.py` |
| Audience act | `projects/impression_management_standard/audience_act_component.py` |
| Game master | `concordia/prefabs/game_master/impression_management_pe.py` |
| IMPE logic | `concordia/components/agent/impression_management_pe.py` |
| Data extraction | `data_extraction.py` |
| Saving results | `results.py` |
| Setup | `setup.py` |
| Detailed component order & flow | `projects/impression_management_standard/docs/component_architecture.md` |

---

## 12. Summary Table: Where Things Live

| What | Where |
|------|--------|
| Run loop, traits resolution, extraction, saving | `main.py` |
| ConversationConfig, TurnLog | `models.py` |
| Prefabs + instances for sim | `simulation_config.py` |
| Norms, traits, role presets, banks, defaults | `constants.py` |
| Actor (PF, reflection, IMPE act, optional wrappers) | `impression_management_actor.Entity` |
| Audience (evaluate on observe, simple act) | `simple_audience_prefab.Entity` |
| Audience “act” = return stored response | `audience_act_component.SimpleAudienceActComponent` |
| Turn order (actor → audience) | `impression_management_pe.GameMaster` |
| Memory, PF, evaluation, reflection, act, norms, traits, world | `impression_management_pe.py` |
| PE/IMPE data classes | `pe_conversation` + `impression_management_pe` |

This summarizes the components and their roles in the impression standard game.
