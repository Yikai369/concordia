# CLI Arguments Guide — Impression Management PE (Standard)

This guide lists all command-line arguments for the standard impression management experiment.

**Entry point:** `python projects/impression_management_standard/main.py [OPTIONS]`

**Environment:** For OpenAI, set `OPENAI_API_KEY`. For local (Ollama), ensure the model is running.

---

## Quick reference

| Category | Key flags |
|----------|-----------|
| **Simulation** | `--turns`, `--window`, `--seed`, `--save_dir` |
| **Model** | `--model`, `--temperature`, `--top_p`, `--llm_type`, `--local_model` |
| **Agents** | `--actor_name`, `--audience_name` |
| **Identity & context** | `--no_traits`, `--no_audience_norms`, `--no_context`, `--interview_role_preset`, `--no_question_bank`, `--no_experience_bank`, `--actor_has_norms` |
| **Perception / world** | `--no_instructions`, `--no_self_perception`, `--enable_situation_perception`, `--enable_person_by_situation`, `--no_world_building`, `--no_interview_context`, `--no_full_2a25` |
| **Self-assessment** | `--enable_self_assessment`, `--consistency_threshold`, `--disable_revision` |
| **Logging & output** | `--enable_info_flow_logging`, `--enable_simplified_log`, `--save_component_logs`, `--no_plots`, `--pretty_trace`, `--outfile` |
| **Experimental** | `--use_option_space`, `--use_memory_check`, `--use_trait_paragraph`, `--enable_question_checks` |

---

## 1. Simulation

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--turns` | int | `2` | Number of dialogue turns (each turn = actor acts, then audience acts). |
| `--window` | int | `3` | Recent K turns to condition on (conversation / PE history window). |
| `--seed` | int | `7` | Random seed for reproducibility (e.g. trait scores). |
| `--save_dir` | str | (timestamped) | Output directory; if omitted, a timestamped dir under `./temp/` is used. |
| `--outfile` | str | `pe_conversation_log.json` | Base name for the conversation log JSON (written inside `save_dir`). |

---

## 2. Model

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `gpt-4o-mini` | OpenAI model name (used when `--llm_type openai`). |
| `--temperature` | float | `0.2` | Sampling temperature. |
| `--top_p` | float | `0.9` | Top-p nucleus sampling. |
| `--llm_type` | str | `openai` | `openai` or `local`. |
| `--local_model` | str | `llama3.1:8b` | Local model name (e.g. Ollama); used when `--llm_type local`. |

---

## 3. Agents

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--actor_name` | str | `John` | Actor (interviewee) name. |
| `--audience_name` | str | `Jane` | Audience (interviewer) name. |

---

## 4. Identity and context

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no_traits` | flag | False | Disable personality traits for both actor and audience. |
| `--traits_file` | str | None | Load traits from Excel (.xlsx) or CSV with columns "name" and "assertion". Ignored if `--no_traits`. |
| `--no_audience_norms` | flag | False | Disable cultural norms for the audience. |
| `--no_context` | flag | False | Disable interview context (role, question/experience banks). |
| `--interview_role_preset` | str | `product_manager` | Preset for role text and optional banks: e.g. `product_manager`, `customer_service`. |
| `--no_question_bank` | flag | False | Do not append question bank to interviewer (audience) context. |
| `--no_experience_bank` | flag | False | Do not append experience bank to interviewee (actor) context. |
| `--actor_has_norms` | flag | False | Give the actor the same cultural norms as the audience. |

---

## 5. Perception and world-building

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--no_instructions` | flag | False | Disable Instructions component (role-playing context). |
| `--no_self_perception` | flag | False | Disable SelfPerception ("who am I?"). |
| `--enable_situation_perception` | flag | **True** | Enable SituationPerception ("what situation am I in?"). Use `--no_situation_perception` to disable. |
| `--enable_person_by_situation` | flag | **True** | Enable PersonBySituation ("what would I do?"). Requires `--enable_situation_perception`. Use `--no_person_by_situation` to disable. |
| `--no_situation_perception` | flag | False | Disable SituationPerception (overrides default). |
| `--no_person_by_situation` | flag | False | Disable PersonBySituation (overrides default). |
| `--no_world_building` | flag | False | Disable 2A25 world-building (Cadens/Riffers narrative). |
| `--no_interview_context` | flag | False | Disable interview-specific context in world-building. |
| `--no_full_2a25` | flag | False | Use minimal generic world text instead of full 2A25 narrative. |

---

## 6. Self-assessment

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_self_assessment` | flag | **True** | Enable self-assessment (consistency check and optional revision). Use `--no_self_assessment` to disable. |
| `--consistency_threshold` | float | `0.7` | Minimum consistency score (0–1) to accept without revision. |
| `--no_self_assessment` | flag | False | Disable self-assessment (overrides default). |
| `--disable_revision` | flag | False | Disable revision of inconsistent responses; only log assessments. |

---

## 7. Logging and output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enable_info_flow_logging` | flag | False | Log all LLM prompts and responses (information flow history). |
| `--enable_simplified_log` | flag | False | Write a simplified, human-readable info-flow log. Requires `--enable_info_flow_logging`. |
| `--simplified_log_format` | str | `compact` | Format for simplified log: `compact`, `markdown`, or `text`. |
| `--save_component_logs` | flag | False | Save Concordia component-level logs to `component_logs.json`. |
| `--no_plots` | flag | False | Disable generating plots (e.g. pe.png, delta_I.png). |
| `--pretty_trace` | flag | False | Print a prettier conversation trace to stdout. |

---

## 8. Experimental / optional features

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_option_space` | flag | False | Generate 4 response options per turn and choose one (2 LLM calls per turn per agent). |
| `--use_memory_check` | flag | False | Inject full-conversation summary into audience and actor prompts (1 extra LLM call per turn). |
| `--use_trait_paragraph` | flag | False | Use one LLM-generated paragraph per agent for personality (adds 1 LLM call per agent). |
| `--enable_question_checks` | flag | False | After the run, ask the model to summarize situation and personality per agent (2 LLM calls per agent). |

---

## Example commands

**Minimal (2 turns, default model):**
```bash
python projects/impression_management_standard/main.py --turns 2
```

**With self-assessment and component logs:**
```bash
python projects/impression_management_standard/main.py --turns 4 --enable_self_assessment --save_component_logs
```

**Minimal identity (no traits, no norms, no context):**
```bash
python projects/impression_management_standard/main.py --turns 2 --no_traits --no_audience_norms --no_context
```

**Full logging (info flow + simplified + component logs):**
```bash
python projects/impression_management_standard/main.py --turns 2 --enable_info_flow_logging --enable_simplified_log --save_component_logs
```

**Local model (Ollama):**
```bash
python projects/impression_management_standard/main.py --turns 2 --llm_type local --local_model llama3.1:8b
```

**Custom output directory and seed:**
```bash
python projects/impression_management_standard/main.py --turns 3 --save_dir ./my_run --seed 42
```

---

## Validation notes

- `--consistency_threshold` must be between 0.0 and 1.0.
- `--enable_simplified_log` requires `--enable_info_flow_logging`; the parser will error otherwise.
- For OpenAI, `OPENAI_API_KEY` must be set (env or `.env` in the project directory).
