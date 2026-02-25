# IMPE Full Turn: Component and Information Flow Schematic

This document describes the components involved in a full turn for the **actor** and **audience** in the standard IMPE simulation, and how information flows through them.

---

## 1. Simulation loop (one step)

The sequential engine runs one **step** as follows:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP N                                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. OBSERVE (all entities in parallel)                                      │
│     • Game master: make_observation(game_master, entity) → observation       │
│     • For each entity: entity.observe(observation)                           │
│       → PRE_OBSERVE (all context components) → POST_OBSERVE → UPDATE        │
│                                                                             │
│  2. NEXT_ACTING                                                             │
│     • Game master picks next_entity and action_spec                          │
│                                                                             │
│  3. ACT (single entity)                                                     │
│     • next_entity.act(action_spec)                                           │
│       → PRE_ACT (all context components) → ACT (act_component) →             │
│         POST_ACT → UPDATE                                                    │
│     • Raw action (e.g. "John -- \"...\"" or "Jane: DIALOGUE: ... BODY: ...") │
│                                                                             │
│  4. RESOLVE                                                                 │
│     • game_master.observe(putative_event)                                    │
│     • game_master.act(resolve) → result                                      │
│     • game_master.observe(EVENT_TAG + result)                                │
│     • Resolved event becomes the next thing all entities will observe        │
└─────────────────────────────────────────────────────────────────────────────┘
```

So in one step: **everyone observes** (current resolved event / situation), then **one entity acts**, then the game master **resolves** that action into the next event.

---

## 2. Actor (e.g. John) – components and information flow

### 2.1 Actor components (standard / impression_management_actor prefab)

| Order | Component key / type | Role |
|------|----------------------|------|
| 1 | Instructions | Experimental/role context |
| 2 | SelfPerception | "What kind of person is {name}?" |
| 3 | SituationPerception | "What kind of situation is {name} in?" |
| 4 | PersonBySituation | "What would a person like {name} do in this situation?" |
| 5 | Memory (AssociativeMemory) | General episodic memory |
| 6 | IMPEMemoryComponent | Goal, _conversation, PF state, evaluations, reflection |
| 7 | ObservationToMemory | Stores observations in associative memory |
| 8 | IMPEActorParticleFilterComponent | PF over I (goal attainment); observes audience feedback |
| 9 | IMPEReflectionComponent | Reflection text (e.g. pre_act, after PF update) |
| 10 | WorldContext (optional) | World/setting text |
| 11 | CulturalNorms (optional) | Norms for behaviour |
| 12 | PersonalityTraits (optional) | Trait scores/text |
| — | **Act component** | IMPEActComponent (or IMPESelfAssessmentComponent wrapping it) |

### 2.2 Actor: when he OBSERVES

Observation is typically the **resolved event** from the game master, e.g.  
`Event: Jane -- DIALOGUE: ... BODY: ...` (audience’s reply) or `Event: John -- "..."` (own prior line).

```
  observation (string)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PRE_OBSERVE (all context components, in parallel)               │
  ├─────────────────────────────────────────────────────────────────┤
  │ • ObservationToMemory.pre_observe(obs)                           │
  │     → stores observation in AssociativeMemory                    │
  │                                                                 │
  │ • IMPEActorParticleFilterComponent.pre_observe(obs)               │
  │     → parse "Audience said: \"...\"", "Body language: \"...\""   │
  │     → set _last_audience_text, _last_audience_body               │
  │     → IMPEMemory: add_utterance(turn, 'interviewer'|'listener',  │
  │         _last_audience_text, _last_audience_body)  [full conv]   │
  │                                                                 │
  │ • IMPEReflectionComponent: no-op in pre_observe                  │
  │ • Other components: no pre_observe or no-op                     │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ POST_OBSERVE                                                     │
  ├─────────────────────────────────────────────────────────────────┤
  │ • IMPEActorParticleFilterComponent.post_observe()                 │
  │     → IMPEMemory.get_goal(), get_recent_conversation()           │
  │     → PF: predict → measurement from audience text/body → update │
  │     → memory.set_pf_state(particles, weights)                    │
  │     → memory.add_evaluation_record(...) for PE etc.              │
  │ • IMPEReflectionComponent: no-op in post_observe                  │
  └─────────────────────────────────────────────────────────────────┘
```

So for the **actor**, when he observes the audience’s reply: **pre_observe** parses it, adds the audience’s utterance to IMPE `_conversation`, then **post_observe** runs the particle filter update and stores PF/evaluation state.

### 2.3 Actor: when he ACTS

The action spec asks what the actor will say next (e.g. `{name} -- "..."`).

```
  action_spec
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PRE_ACT (all context components, in parallel)                    │
  ├─────────────────────────────────────────────────────────────────┤
  │ • Instructions, SelfPerception, SituationPerception,             │
  │   PersonBySituation: run pre_act; their output is now included  │
  │   in the action prompt via IMPEActComponent._get_context_block  │
  │   (see ACT phase).                                               │
  │                                                                 │
  │ • IMPEMemoryComponent.pre_act()                                  │
  │     → goal, recent conversation, (optional) conversation        │
  │       summary (memory check)                                     │
  │                                                                 │
  │ • IMPEActorParticleFilterComponent: no pre_act contribution      │
  │                                                                 │
  │ • IMPEReflectionComponent.pre_act()                             │
  │     → builds reflection using get_recent_conversation(), PF      │
  │       state, goal; stores in memory; reflection text appears     │
  │       in act prompt via memory.get_recent_reflections()          │
  │                                                                 │
  │ • IMPEActComponent: no pre_act (builds the action prompt in      │
  │   get_action_attempt using memory + _get_prompt_header +         │
  │   _get_context_block(context) for Instructions/Self/etc.)       │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ ACT (act_component only)                                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ • IMPEActComponent.get_action_attempt(context, action_spec)     │
  │     → Builds prompt from:                 │
  │       - _get_prompt_header(): WorldContext, CulturalNorms,      │
  │         PersonalityTraits (via get_component)                    │
  │       - _get_context_block(context): Instructions, SelfPerception, │
  │         SituationPerception, PersonBySituation (pre_act context)│
  │       - IMPEMemory: goal, recent conversation, PF history,       │
  │         recent reflections, (optional) conversation summary    │
  │     → model.sample_text(prompt) → actor’s dialogue line         │
  │     → IMPEMemory.add_utterance(...) [store own line]             │
  │     → return e.g. John -- "..."                                 │
  └─────────────────────────────────────────────────────────────────┘
```

So when the **actor** acts: **WorldContext**, **CulturalNorms**, **PersonalityTraits**, **Instructions**, **SelfPerception**, **SituationPerception**, **PersonBySituation** (when present), **IMPEMemory**, and **IMPEReflectionComponent** (via memory) all feed into the action prompt.

---

## 3. Audience (e.g. Jane) – components and information flow

### 3.1 Audience components (simple_audience_prefab)

| Order | Component key / type | Role |
|------|----------------------|------|
| 1 | Instructions | Role-playing context |
| 2 | SelfPerception | "What kind of person is {name}?" |
| 3 | SituationPerception | Optional |
| 4 | PersonBySituation | Optional |
| 5 | Memory (AssociativeMemory) | General episodic memory |
| 6 | IMPEMemoryComponent | Goal, _conversation, evaluations |
| 7 | ObservationToMemory | Stores observations in associative memory |
| 8 | IMPEAudienceEvaluationComponent | Evaluates actor, generates reply |
| 9 | WorldContext (optional) | World/setting |
| 10 | CulturalNorms (optional) | Norms |
| 11 | PersonalityTraits (optional) | Trait scores/text |
| — | **Act component** | SimpleAudienceActComponent |

### 3.2 Audience: when she OBSERVES

Observation is typically the **resolved event** with the **actor’s** line, e.g.  
`Event: John -- "..."`.

```
  observation (string)
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PRE_OBSERVE                                                      │
  ├─────────────────────────────────────────────────────────────────┤
  │ • ObservationToMemory.pre_observe(obs)                           │
  │     → stores in AssociativeMemory                               │
  │                                                                 │
  │ • IMPEAudienceEvaluationComponent.pre_observe(obs)                │
  │     → parse "Actor said: \"...\"", "Body language: \"...\""      │
  │     → set _last_actor_text, _last_actor_body                     │
  │     → IMPEMemory: add_observation(turn, observed_from=...,       │
  │         text, body)                                              │
  │     → IMPEMemory: add_utterance(turn, 'Actor',                   │
  │         _last_actor_text, _last_actor_body)  [full conversation] │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ POST_OBSERVE                                                     │
  ├─────────────────────────────────────────────────────────────────┤
  │ • IMPEAudienceEvaluationComponent.post_observe()                  │
  │     → rate actor (I_t) from _last_actor_text / _last_actor_body  │
  │     → build "Recent conversation" from memory.format_conversation │
  │       (now includes actor’s line from pre_observe)                │
  │     → optional: memory_check summary injected                   │
  │     → generate reply (DIALOGUE + BODY) via model                 │
  │     → memory.add_utterance(turn, self.get_entity().name, dlg, body) │
  │       [audience’s own reply into _conversation]                   │
  │     → memory.add_evaluation_record(turn, I_t, utt)              │
  │     → return summary string (for logging)                        │
  └─────────────────────────────────────────────────────────────────┘
```

So for the **audience**, when she observes the actor’s line: **pre_observe** parses it, adds the actor’s utterance to IMPE `_conversation`, and **post_observe** evaluates the actor, generates her reply, and adds both her utterance and the evaluation to IMPE memory.

### 3.3 Audience: when she ACTS

The game master asks the audience what they say/do next. The audience does **not** generate a new reply in act(); she **returns** the reply that was already generated in **post_observe**.

```
  action_spec
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ PRE_ACT (all context components)                                 │
  ├─────────────────────────────────────────────────────────────────┤
  │ • Instructions, SelfPerception, ... → context text               │
  │ • SimpleAudienceActComponent: _make_pre_act_value() → ''          │
  └─────────────────────────────────────────────────────────────────┘
       │
       ▼
  ┌─────────────────────────────────────────────────────────────────┐
  │ ACT (act_component only)                                         │
  ├─────────────────────────────────────────────────────────────────┤
  │ • SimpleAudienceActComponent.get_action_attempt(...)              │
  │     → IMPEMemory.get_recent_conversation()                       │
  │     → find most recent utterance where speaker == audience name │
  │     → return "DIALOGUE: ...\nBODY: ..."                          │
  │   (fallback: get from evaluation record’s utterance)            │
  └─────────────────────────────────────────────────────────────────┘
```

So the **audience**’s act path only **returns** the stored reply; all evaluation and reply generation happened in **post_observe**.

---

## 4. Information flow summary

| Who    | When     | Incoming info         | Components that use it        | What gets written to IMPE memory |
|--------|----------|------------------------|--------------------------------|-----------------------------------|
| Actor  | Observe  | Event: Audience said … | ActorPF (pre), then ActorPF (post) | add_utterance(audience); PF state; evaluation record |
| Actor  | Act      | Context (goal, conv, reflection) | IMPEActComponent              | add_utterance(actor’s own line)   |
| Audience | Observe | Event: Actor said …    | AudienceEval (pre), then (post) | add_utterance(Actor); add_utterance(audience); evaluation record |
| Audience | Act     | —                      | SimpleAudienceActComponent     | (none; reads from memory)        |

- **Actor** IMPE `_conversation`: gets **actor’s own lines** in **act()** and **audience’s lines** in **pre_observe()** (IMPEActorParticleFilterComponent).
- **Audience** IMPE `_conversation`: gets **actor’s lines** in **pre_observe()** (IMPEAudienceEvaluationComponent) and **audience’s own lines** in **post_observe()** (IMPEAudienceEvaluationComponent).

So both sides now have the **full dialogue** in IMPE memory for “Recent conversation” and for the memory-check summary.

---

## 5. Diagram: one full exchange (actor speaks, then audience replies)

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Game master                                                              │
  │  • Resolved event: "Event: John -- \"...\""                                │
  └──────────────────────────────────────────────────────────────────────────┘
       │ make_observation(John)     │ make_observation(Jane)
       ▼                           ▼
  ┌─────────────┐            ┌─────────────┐
  │ John (actor)│            │ Jane (aud.) │
  │ observe(…)  │            │ observe(…)  │
  │ • ObsToMem  │            │ • ObsToMem   │
  │ • ActorPF   │            │ • AudEval    │
  │   pre: —    │            │   pre: add   │
  │   post: PF  │            │     Actor    │
  │   update    │            │     utt     │
  └─────────────┘            │   post: I_t,│
       │                     │   reply,    │
       │                     │   add_utt   │
       │                     └─────────────┘
       │ next_acting → e.g. Jane
       │
       │                     ┌─────────────┐
       │                     │ Jane .act() │
       │                     │ return      │
       │                     │ DIALOGUE +  │
       │                     │ BODY        │
       │                     └─────────────┘
       │                            │
       │  resolve("Jane: DIALOGUE... BODY...") → "Event: Jane -- ..."
       │
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  Next step: everyone observes "Event: Jane -- ..."                        │
  │  John: ActorPF pre_observe → add_utterance(Jane's reply); post → PF      │
  │  Jane: AudEval pre_observe → (actor didn’t speak this step); post —       │
  │  next_acting → John → John.act() → add_utterance(John's line) → resolve   │
  └──────────────────────────────────────────────────────────────────────────┘
```

This matches the standard IMPE loop: one step = all observe (including full-dialogue updates to IMPE memory), one entity acts, then resolve.
