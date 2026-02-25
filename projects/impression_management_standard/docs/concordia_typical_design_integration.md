# Integration with Concordia Typical Action Design

This document specifies whether the current IMPE (Impression Management with Prediction Error) implementation can be integrated with Concordia’s typical action-generation design, and what that would require.

---

## 1. Concordia Typical Design (Summary)

### 1.1 Orchestration

- The **entity agent** runs a fixed act cycle: **PRE_ACT** → **ACT** → **POST_ACT** → **UPDATE**.
- All components’ `pre_act(action_spec)` are called in parallel; their string outputs are collected into a **context mapping** (component name → string).
- The **acting component** is the only one that decides the action: `get_action_attempt(contexts, action_spec) → str`.
- So: context is supplied by pre_act; the act component consumes it and returns one action string.

### 1.2 Typical Acting Component: ConcatActComponent

- **Context**: Concatenates pre_act outputs (optionally in a fixed `component_order`), newline-separated. No other state.
- **Prompt**: One “statement” (the concatenated context) plus one “call to action” from `action_spec.call_to_action`.
- **LM usage**: A single call via `InteractiveDocument`:
  - **FREE**: `prompt.open_question(call_to_action, ...)` → return (optionally entity name + answer).
  - **CHOICE**: `prompt.multiple_choice_question(question, answers)` → return chosen option.
  - **FLOAT**: `prompt.open_question(...)` then parse float.
- **Return**: The model’s answer string (no parsing of structured blocks, no side effects to memory inside the act component).

### 1.3 Contract

- **ActingComponent** interface: `get_action_attempt(context: ComponentContextMapping, action_spec: ActionSpec) -> str`.
- **action_spec**: `call_to_action` (format string with `{name}`) and `output_type` (FREE, CHOICE, FLOAT, etc.).
- The engine/game master expects a string that matches the action spec (e.g. free text or choice label).

---

## 2. Current IMPE Implementation (Summary)

### 2.1 Orchestration

- Uses the **same** entity flow: PRE_ACT → act component’s `get_action_attempt(contexts, action_spec)` → POST_ACT → UPDATE.
- So at the **orchestration** level, IMPE is already integrated: it is a drop-in `ActingComponent`.

### 2.2 IMPEActComponent Behavior

- **Context usage**: Only a **subset** of `contexts`: keys in `context_keys_for_prompt` (e.g. Instructions, SelfPerception, SituationPerception, PersonBySituation). Combined into a single “Identity and situation” block. Other pre_act keys are ignored for the prompt.
- **Additional inputs** (not from pre_act): IMPE memory (goal, conversation, I_hat/pf history, reflections), optional world context, cultural norms, personality traits. These are fetched directly from other components via `get_entity().get_component(...)`.
- **Prompt**: One large hand-built prompt (header + context block + goal + conversation + I_hat + reflections + strict output instructions).
- **Output format**: Model is asked to output exactly:
  - `DIALOGUE: <one sentence>`
  - `BODY: <brief body language phrase>`
- **LM usage**: `model.sample_text(prompt)` (or, if `use_option_space`, two calls: options then choice). No `InteractiveDocument`; no use of `action_spec.call_to_action` for the main utterance (only the returned string is formatted for the game master).
- **Parsing**: Regex to extract `DIALOGUE:` and `BODY:` from raw output; fallback if parsing fails.
- **Side effects**: Writes to IMPE memory: `memory.add_utterance(...)`, `memory.add_action(...)` (unless `skip_memory_update`).
- **Return**: `f'{entity_name} -- "{text}"'` (dialogue only in the string; body is stored but not in the returned action string).

### 2.3 IMPESelfAssessmentComponent

- Wraps another `ActingComponent` (e.g. IMPEActComponent). Calls base `get_action_attempt(..., skip_memory_update=True)`, then runs an assessment/revise step with a second prompt, then updates memory and returns. So: same interface, extra step and logic.

---

## 3. Integration Analysis

### 3.1 What Is Already Aligned

| Aspect | Concordia typical | IMPE current | Aligned? |
|--------|-------------------|---------------|----------|
| Entity act flow | PRE_ACT → get_action_attempt → POST_ACT → UPDATE | Same | Yes |
| ActingComponent interface | get_action_attempt(contexts, action_spec) → str | Same | Yes |
| Use of pre_act | Context = concatenation of pre_act outputs | Context block from selected pre_act keys | Partially (subset, not full concat) |
| Return type | Single string action | Single string `name -- "text"` | Yes |

So: **orchestration and interface are already compatible**. IMPE does not need to change to “plug in” to the entity agent; it already does.

### 3.2 Where IMPE Diverges from Typical Design

| Aspect | Typical (e.g. ConcatAct) | IMPE | Integration impact |
|--------|---------------------------|------|--------------------|
| **Call-to-action** | `action_spec.call_to_action` is the main question | Not used for content; prompt is fixed (“Produce a short utterance… DIALOGUE: … BODY: …”) | Game master / engine cannot drive IMPE’s “task” via action_spec; IMPE is goal/memory-driven. |
| **LM API** | InteractiveDocument (statement + open_question / multiple_choice_question) | Raw `model.sample_text(prompt)` | Different pattern; no structural change needed to integrate, but would require refactor to use InteractiveDocument if we want one API. |
| **Output shape** | One free-form or choice answer, no parsing | Structured DIALOGUE + BODY, regex parsing, fallbacks | Typical design has no DIALOGUE/BODY; IMPE needs this for body language and memory. |
| **Memory writes in act** | None | add_utterance, add_action | Typical act components are stateless w.r.t. memory; IMPE updates conversation state in the same turn. |
| **Context source** | Only pre_act mapping | pre_act (subset) + IMPEMemory + WorldContext + CulturalNorms + PersonalityTraits | IMPE pulls more from the entity than “contexts” alone. |
| **action_spec.output_type** | Drives FREE vs CHOICE vs FLOAT and LM call | Effectively ignored (always generating one utterance) | IMPE does not support multiple output types; it’s dialogue-only. |

### 3.3 Can We “Integrate” (Match Typical Design Closely)?

**Option A: Make IMPE behave like ConcatAct**

- **Would require**: (1) Using only concatenated pre_act as context (or a single “statement”); (2) using `action_spec.call_to_action` as the only question; (3) one `open_question` call, no DIALOGUE/BODY structure; (4) no memory writes inside `get_action_attempt` (move to POST_ACT or a separate component).
- **Effect**: We would lose IMPE’s current behavior: goal-driven prompts, I_hat and reflections, body language, and in-turn memory updates. Not recommended if we want to keep IMPE’s semantics.

**Option B: Keep IMPE as-is and document the contract**

- **Current state**: IMPE is already a valid `ActingComponent` and works within the same entity and phase flow.
- **Integration**: “Integrated” in the sense that it uses the same lifecycle and interface; it intentionally diverges in prompt design, output format, and memory updates.
- **Recommendation**: This is the practical integration: **keep the current implementation**, and document that IMPE is an **alternative** acting pattern (memory- and goal-driven, structured output, in-turn memory) rather than a ConcatAct-style (context-only, call-to-action-driven, single-answer) component.

**Option C: Hybrid – use typical design where possible**

- Use **pre_act** for all “context” (identity, situation, instructions): already done via `_get_context_block`.
- Optionally use **InteractiveDocument**: e.g. `statement(header + context_block + body_text)`, then `open_question("Produce one utterance... Output format: DIALOGUE: ... BODY: ...")` so the LM call pattern matches ConcatAct, while keeping DIALOGUE/BODY and parsing. This would be a refactor for consistency, not required for correctness.
- Keep **memory writes** and **goal/I_hat/reflections** inside IMPE; they are essential to the design and have no counterpart in ConcatAct.
- **action_spec**: Could optionally append `action_spec.call_to_action` to the prompt (e.g. as an extra instruction) when provided, so the game master can nudge the task without changing IMPE’s core prompt. Not required for integration.

---

## 4. Conclusion and Recommendations

### 4.1 Can we integrate?

- **Yes, in the sense already achieved**: IMPE is integrated with Concordia’s **orchestration and component interface**. It runs in the same entity agent, receives the same `contexts` and `action_spec`, and returns a string. No change is required for it to “work” with the rest of Concordia.
- **No, in the sense of making IMPE identical to ConcatAct**: That would mean removing goal/memory-driven prompts, DIALOGUE/BODY structure, and in-turn memory updates, which would break the IMPE design.

### 4.2 Recommended stance

1. **Treat IMPE as a valid, alternative acting component** that follows the **contract** (get_action_attempt(contexts, action_spec) → str) and the **lifecycle** (PRE_ACT → ACT → POST_ACT → UPDATE), but deliberately uses a different **internal** design (memory- and goal-driven, structured output, side effects).
2. **Document the contract** for consumers:
   - IMPE largely ignores `action_spec.call_to_action` and `output_type`; the “task” is defined by the IMPE goal and prompt.
   - Return format is `{entity_name} -- "{dialogue_text}"`; body language is stored in IMPE memory but not in the returned string.
   - If the game master or engine expects a specific call_to_action or output_type, IMPE may not satisfy it; use ConcatAct (or another component) for that entity instead.
3. **Optional refinements** (if we want closer alignment without losing behavior):
   - Use `InteractiveDocument` for the main LM call (statement + open_question with format instructions) so the call pattern matches ConcatAct.
   - Optionally incorporate `action_spec.call_to_action` into the prompt when non-empty, for flexibility.

### 4.3 Summary table

| Question | Answer |
|----------|--------|
| Does IMPE plug into the same entity/phase flow? | Yes. |
| Does it implement the ActingComponent interface? | Yes. |
| Can we switch to ConcatAct-style without losing IMPE behavior? | No. |
| Should we change IMPE to match ConcatAct exactly? | No. |
| Is the current implementation “integrated” with Concordia typical design? | Yes, at the orchestration and interface level; no, at the internal prompt/LM/memory level, by design. |

This document can be updated if we add a hybrid (e.g. InteractiveDocument or optional call_to_action) or new acting components that blend both patterns.
