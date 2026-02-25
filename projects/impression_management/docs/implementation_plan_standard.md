# Step-by-Step Plan: Implement Prototype Features in Standard Version

Based on [prototype_vs_standard_comparison.md](prototype_vs_standard_comparison.md), this document outlines a phased plan to add prototype-only features to `impression_management_standard` while keeping the Concordia architecture and configurable design.

---

## Summary: What to Add

| Feature | Priority | Scope | Notes |
|--------|----------|--------|--------|
| Spreadsheet trait loading | High | Optional | Load traits from Excel; keep fixed list as default |
| Trait paragraph generation | High | Optional | LLM paragraph from traits; optional mode |
| Trait paragraphs in JSON output | High | With paragraph mode | Save `actor_traits`, `audience_traits` when used |
| Interview role presets | Medium | Config | e.g. Product Manager, Customer Service Agent |
| Question/experience banks | Medium | Optional | Configurable question bank (interviewer), experience bank (interviewee) |
| Option: cultural norms for actor | Low | Config flag | `--actor_has_norms` |
| Option space generation | Low | Experimental | Generate 4 options, choose one; behind flag |
| Question checks | Low | Optional | Verify personality/context; behind flag |

**Out of scope (by design):** Switching to manual loop, adding debug prints, changing field names to actor/audience, or unconditional self-reflection (standard’s threshold-based approach is preferred).

---

## Phase 1: Spreadsheet Trait Loading

**Goal:** Support loading personality traits from an Excel file as an alternative to the fixed list.

### Step 1.1 – Trait loader utility

- **File:** `projects/impression_management_standard/utils.py` (or new `trait_loader.py`).
- **Actions:**
  1. Add optional dependency: `openpyxl` or `pandas` for Excel (document in README).
  2. Implement `load_traits_from_spreadsheet(file_path: str) -> list[PersonalityTrait]`. Use `PersonalityTrait` from `concordia.components.agent.impression_management_pe` (or import via `projects.impression_management_standard.constants`, which re-exports it).
  3. `PersonalityTrait` is a dataclass with `name: str` and `assertion: str`. Map spreadsheet columns to these (e.g. one column = name, another = assertion; or first row = names).
  4. Return list compatible with `PersonalityTraitsComponent` and `constants.ALL_TRAITS` (same type).
- **Edge cases:** Empty file, missing columns, non-numeric sheet; return empty list or raise clear error.

### Step 1.2 – Config and CLI

- **File:** `projects/impression_management_standard/models.py`: Add `traits_file: str | None = None` to the `ConversationConfig` dataclass.
- **File:** `projects/impression_management_standard/config.py`: Add CLI argument `--traits_file` (optional path); when building `ConversationConfig`, set `traits_file` from args. If `--no_traits` is set, ignore `traits_file` when loading traits.
- **File:** `projects/impression_management_standard/main.py`: After parsing config, if `cfg.traits_file` is set, call `load_traits_from_spreadsheet(cfg.traits_file)` and pass the returned list into simulation config; otherwise use `constants.ALL_TRAITS` as today.

### Step 1.3 – Simulation config wiring

- **File:** `projects/impression_management_standard/main.py`: Before calling `create_simulation_config`, resolve traits: if `cfg.no_traits`, use `None`; else if `cfg.traits_file`, call `load_traits_from_spreadsheet(cfg.traits_file)` and use the result; else use `constants.ALL_TRAITS`. Pass this resolved `traits` list into `create_simulation_config` (e.g. add an optional parameter `traits_override: list | None = None`; when provided, use it instead of reading from constants inside simulation_config).
- **File:** `projects/impression_management_standard/simulation_config.py`: Accept the traits list from the caller (or fall back to `constants.ALL_TRAITS` when no override). Pass that list into instance params for both actor and audience. Keep using `utils.generate_trait_scores()` for score-based mode when traits are loaded from spreadsheet (scores still 0–3; decide if spreadsheet can optionally supply scores later).

### Step 1.4 – Tests and docs

- Add a small test Excel (or CSV if you add CSV support) and a unit test for `load_traits_from_spreadsheet`.
- Update README: how to use `--traits_file`, expected Excel format, optional dependency.

---

## Phase 2: Trait Paragraph Generation (Optional Mode)

**Goal:** Optional “paragraph mode”: LLM turns trait assertions into a single narrative paragraph used in prompts; standard score-based mode remains default.

### Step 2.1 – Paragraph generation in framework

- **Location:** Either a new component in `concordia/components/agent/impression_management_pe.py` or an optional path inside `PersonalityTraitsComponent`.
- **Actions:**
  1. Add a component (e.g. `TraitParagraphComponent`) or a mode in `PersonalityTraitsComponent` that:
     - Takes the same trait list (or loaded-from-spreadsheet list).
     - Calls the language model once with a fixed prompt: “Write a short paragraph describing this person from these statements: …” (list assertions).
     - Caches the result (e.g. in component state or in entity-scoped memory) so it’s not recomputed every turn.
  2. Ensure this runs only at “initialization” (e.g. first time the entity acts or in a dedicated init step), not every turn.
  3. Expose a method like `get_traits_text() -> str` that returns either:
     - Score-based text (current behavior), or  
     - The generated paragraph (when paragraph mode is on).

### Step 2.2 – Config and prefab wiring

- **Config:** Add `use_trait_paragraph: bool = False` to `ConversationConfig` (models.py) and CLI `--use_trait_paragraph` (config.py).
- **Instance params:** In `simulation_config.py`, pass `use_trait_paragraph` in the params dict for both actor and audience instances. The Concordia prefabs (`concordia.prefabs.entity.impression_management_actor`, `projects.impression_management_standard.simple_audience_prefab`) receive these params and pass them into `PersonalityTraitsComponent` (or the new component). When `use_trait_paragraph` is True, the component generates and uses the paragraph; else use current score-based text.

### Step 2.3 – Memory / extraction for output

- **Goal:** When paragraph mode is used, trait paragraphs must be available for JSON output.
- **Option A – From components:** In `data_extraction.py` (or wherever you build the “run summary”), get the personality component(s) from actor and audience entities and call a method like `get_traits_text()` or `get_trait_paragraph()`; store in a structure used by results (e.g. `actor_trait_paragraph`, `audience_trait_paragraph`).
- **Option B – From simulation config / run state:** If paragraphs are generated at setup and stored in a run-level structure, pass that into the results saver.
- **Files:** `data_extraction.py`, `results.py`, and possibly a small extension to the model used for “run output” (see Phase 3).

### Step 2.4 – Docs and tests

- Document: when to use paragraph vs score-based; that paragraph adds one LLM call per agent at init; reproducibility note (paragraph varies by run).
- Optional: unit test with mocked LLM returning a fixed paragraph and assert it appears in `get_traits_text()`.

---

## Phase 3: Trait Paragraphs in JSON Output

**Goal:** When trait paragraphs are used, save them in the main JSON output so they match the prototype’s structure.

### Step 3.1 – Output shape

- **File:** `projects/impression_management_standard/models.py` (or equivalent).
- **Actions:**
  1. Define a small structure for “run output” that includes:
     - `actor_traits: str | None`
     - `audience_traits: str | None`
     - `turns: list[TurnLog]` (existing).
  2. Keep backward compatibility: if not using paragraph mode, `actor_traits` and `audience_traits` can be omitted or null; existing consumers that only read `turns` still work.

### Step 3.2 – Populate and save

- **File:** `projects/impression_management_standard/data_extraction.py`: When building the run result, if paragraph mode is on, get actor and audience trait paragraphs from the entities/components (see Phase 2.3) and set `actor_traits` and `audience_traits`. Either return a richer structure (e.g. a run-output object) or extend the extraction API so the caller can pass trait paragraphs into the saver.
- **File:** `projects/impression_management_standard/results.py`: In `save_results`, accept optional `actor_traits: str | None = None` and `audience_traits: str | None = None`. If either is present, write JSON as `{ "actor_traits": ..., "audience_traits": ..., "turns": [ ... ] }`; otherwise keep the current format (a JSON array of turn log objects only) for backward compatibility.

### Step 3.3 – Tests

- Run a short simulation with `--use_trait_paragraph`, then assert the saved JSON has `actor_traits` and `audience_traits` and `turns`.

---

## Phase 4: Interview Role Presets and Question/Experience Banks

**Goal:** Support multiple interview roles (e.g. Product Manager, Customer Service Agent) and optional per-role question banks (interviewer) and experience banks (interviewee).

### Step 4.1 – Role presets in constants

- **File:** `projects/impression_management_standard/constants.py`
- **Actions:**
  1. Keep `DEFAULT_INTERVIEW_ROLE` as Product Manager.
  2. Add e.g. `INTERVIEW_ROLE_PRESETS: dict[str, str]` with keys like `"product_manager"`, `"customer_service"` and values = full role text (responsibilities, evaluation criteria).
  3. Add optional `INTERVIEW_QUESTION_BANKS: dict[str, list[str]]` (e.g. `"customer_service": ["Tell me about your customer service experience", ...]`).
  4. Add optional `INTERVIEW_EXPERIENCE_BANKS: dict[str, list[str]]` (e.g. `"customer_service": ["Experience 1: Managed high-volume frontline support...", ...]`) for the interviewee to draw on.

### Step 4.2 – Config and CLI

- **Config:** Add `interview_role_preset: str = "product_manager"` (or `None` to use current single default). Optional: `interview_question_bank_key: str | None`, `interview_experience_bank_key: str | None` (default from preset).
- **CLI:** e.g. `--interview_role_preset customer_service`; optionally `--no_question_bank`, `--no_experience_bank` to disable banks even when preset supports them.

### Step 4.3 – Use in prompts (world/context)

- **Location:** Wherever interview context and role are injected (e.g. in `concordia/components/agent/impression_management_pe.py` or project-specific prompt builder).
- **Actions:**
  1. Resolve role text from `interview_role_preset` (e.g. from `INTERVIEW_ROLE_PRESETS[preset]`).
  2. For the audience (interviewer) entity: if a question bank exists for the preset, append “You can ask questions such as: …” (or similar) to the context.
  3. For the actor (interviewee) entity: if an experience bank exists, append “You can draw on experiences such as: …” to the context.
  4. Keep all of this behind “interview context” / `no_context` so disabling context disables role and banks.

### Step 4.4 – Simulation config

- **File:** `projects/impression_management_standard/simulation_config.py`
- **Actions:**
  1. Pass `interview_role_preset`, question bank key, experience bank key (or the resolved strings) into actor and audience instance params.
  2. Prefabs/components that build prompts should read these and use the constants above.

### Step 4.5 – Docs

- Document presets and format of question/experience banks; how to add new presets.

---

## Phase 5: Optional “Cultural Norms for Actor”

**Goal:** Configurable option to give the actor the same cultural norms as the audience (as in the prototype).

### Step 5.1 – Config and CLI

- **Config:** Add `actor_has_norms: bool = False`.
- **CLI:** Add `--actor_has_norms`.

### Step 5.2 – Simulation config

- **File:** `projects/impression_management_standard/simulation_config.py`
- **Actions:**
  1. For the actor instance params, set `cultural_norms` to the same value as the audience when `config.actor_has_norms` is True (e.g. use the same `cultural_norms` variable already computed for the audience: `cultural_norms if config.actor_has_norms else None`). Today the actor always gets `cultural_norms: None`.

### Step 5.3 – Docs

- One line in README: “Use `--actor_has_norms` to give the interviewee the same cultural norms as the interviewer.”

---

## Phase 6: Option Space Generation (Experimental)

**Goal:** Optional “option space” mode: generate 4 response options and let the model choose one (for audience response and/or actor utterance).

### Step 6.1 – Where to plug in

- **Audience:** The audience’s reply is produced by `IMPEAudienceEvaluationComponent` in `concordia/components/agent/impression_management_pe.py` (used by `simple_audience_prefab`). Add an optional branch there: instead of one LLM call for the evaluation response, (1) one call to generate 4 options (DIALOGUE + BODY each), (2) parse options, (3) one call to choose one option with reasoning.
- **Actor:** The actor’s utterance is produced by `IMPESelfAssessmentComponent` (act component) in the same file. Add an optional branch there: generate 4 options then choose one.

### Step 6.2 – Implementation steps

1. Add prompts (in code or constants) for “generate 4 options” and “choose one option with brief reasoning,” matching prototype format (numbered list, DIALOGUE/BODY).
2. Add a helper to parse the 4-option response into a list of (text, body) and a helper to parse the choice (e.g. “1”–“4”).
3. Add a config flag e.g. `use_option_space: bool = False` and CLI `--use_option_space`.
4. In the relevant component(s), when `use_option_space` is True, use the two-step flow; otherwise keep current single-call flow.
5. Optionally log the 4 options and the chosen index in component logs or in the run output for analysis.

### Step 6.3 – Docs and tests

- Mark as experimental; document extra LLM cost (2 calls instead of 1 per turn for that agent). Add a simple test with mocked LLM.

---

## Phase 7: Question Checks (Optional)

**Goal:** Optional “question check” step: after each turn (or at end of run), ask the model to summarize “what kind of situation is this?” and “what kind of person are you?” for analysis/debugging.

### Step 7.1 – Design choice

- **Option A – Per-turn:** Like prototype, two extra LLM calls per turn per agent (expensive).
- **Option B – Once per run:** One call per agent at the end: “Summarize the situation” and “Summarize your personality/behavior.” Cheaper and still useful for logging.

Recommendation: start with Option B (once per run).

### Step 7.2 – Implementation

1. Add config `enable_question_checks: bool = False` and CLI `--enable_question_checks`.
2. After the simulation loop, if enabled, use the same language model instance already available in `main.py` to make two LLM calls per entity (situation summary, personality summary), passing each entity’s name and any needed context (e.g. from entity state or conversation history). Collect answers.
3. Store in run output (e.g. `actor_context_summary`, `actor_personality_summary`, and same for audience).
4. In `results.py`, when saving JSON, if question checks were run, include these fields (e.g. top-level or under a `question_checks` object). Extend `save_results` to accept optional question-check fields and write them when present.

### Step 7.3 – Docs

- Document that this is for analysis/debugging and adds 2 LLM calls per agent per run when enabled.

---

## Phase 8: 2A25 World-Building (Align with Prototype)

**Goal:** Ensure the standard’s world-building text matches the prototype’s 2A25 narrative where desired, and is switchable.

### Step 8.1 – Audit current text

- **File:** `concordia/components/agent/impression_management_pe.py`: The 2A25/Cadens/Riffers narrative lives in `WorldContextComponent` (around lines 628–652). Compare that block with the prototype’s full world-building text (see comparison doc).

### Step 8.2 – Optional “full 2A25” mode

- Add config `use_full_2a25_world: bool = True` (or `world_building: str = "2a25"` with options `"2a25"` | `"minimal"`) and CLI.
- **Actions:**
  1. If “full 2A25”, use the prototype’s full narrative (copy into constants or component).
  2. If “minimal”, use a shorter generic alternative (e.g. “You are in an alternative world…” without Cadens/Riffers detail).
  3. Wire the choice into the component that builds the world-building prompt segment.

### Step 8.3 – Docs

- Describe the two modes and when to use each.

---

## Implementation Order and Dependencies

**Strict dependencies:**
- **Phase 1** (spreadsheet traits): No dependencies. Do first so that Phase 2 can use spreadsheet-loaded traits for paragraph mode.
- **Phase 2** (trait paragraph) must be done before **Phase 3** (trait paragraphs in JSON). Do 2 and 3 together; Phase 3 depends on Phase 2.

**Independent phases** (no dependency on each other; order is flexible):
- **Phase 4** (role presets and banks)
- **Phase 5** (actor norms) — one small change; can be done anytime
- **Phase 6** (option space) — experimental, touches Concordia IMPE components
- **Phase 7** (question checks) — simpler (main + results only)
- **Phase 8** (2A25 world-building) — can be done early for narrative parity with prototype

**Suggested order (default):**  
**1 → 2+3 → 4 → 5 → 8 → 6 → 7**

**Alternative orderings:**
- For **narrative parity first**: do Phase 8 right after 2+3: **1 → 2+3 → 8 → 4 → 5 → 6 → 7**.
- For **simpler features before experimental**: do Phase 7 before Phase 6: **1 → 2+3 → 4 → 5 → 8 → 7 → 6**.
- **Quick win**: do Phase 5 right after 2+3: **1 → 2+3 → 5 → 4 → 8 → 6 → 7**.

---

## Files to Touch (Checklist)

| Phase | Files |
|-------|--------|
| 1 | `projects/impression_management_standard/utils.py` or `trait_loader.py`, `models.py`, `config.py`, `main.py`, `simulation_config.py`, README |
| 2 | `concordia/components/agent/impression_management_pe.py` (PersonalityTraitsComponent or new component), project `config.py`, `simulation_config.py`, `data_extraction.py` |
| 3 | `projects/impression_management_standard/models.py` (run output type if needed), `data_extraction.py`, `results.py` |
| 4 | `projects/impression_management_standard/constants.py`, `config.py`, `main.py`, `simulation_config.py`, IMPE prompt builder (in concordia or project) |
| 5 | `models.py`, `config.py`, `main.py`, `simulation_config.py` (all in project) |
| 6 | `concordia/components/agent/impression_management_pe.py` (IMPEAudienceEvaluationComponent, IMPESelfAssessmentComponent), project `config.py`, `main.py` |
| 7 | `models.py`, `config.py`, `main.py`, post-run extraction, `results.py` (all in project) |
| 8 | `concordia/components/agent/impression_management_pe.py` (WorldContextComponent), project `constants.py` or config, `main.py` |

---

## Success Criteria

- All new behavior is behind flags or optional config; default behavior unchanged.
- Existing tests and runs without new flags remain valid.
- README and comparison doc updated to state which prototype features are now available in standard and how to enable them.
