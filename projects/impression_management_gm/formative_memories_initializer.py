"""Formative memory generation and entity initialization utilities."""

import random
import re
import time

from concordia.components import agent as agent_components
from concordia.components.agent import \
    impression_management_pe as impe_components
from concordia.language_model import language_model

from projects.impression_management_gm import constants


class FormativeMemoriesInitializer:
  """Generates and injects formative memories for candidate/interviewer entities."""

  def __init__(self, model: language_model.LanguageModel):
    self._model = model

  def generate_memories(
      self,
      *,
      agent_name: str,
      role: str,
      neurotype: str,
      prompts: list[str],
      count: int,
  ) -> list[str]:
    """Generate 20-50 memory snippets from role-specific prompts."""
    prompt_block = '\n'.join(f'- {p}' for p in prompts)
    out: list[str] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = 6

    while len(out) < count and attempts < max_attempts:
      attempts += 1
      remaining = count - len(out)
      generation_prompt = f"""You are generating formative life memories for an individual.

    Agent name: {agent_name}

This individual lives in a society where the following communication norms are typical:

{constants.CADEN_COMMUNICATION_NORMS}

These norms shape everyday interactions in school, work, and relationships.

IMPORTANT:
- Do NOT explicitly list or restate these norms in the memories
- Instead, let them appear naturally through situations, expectations, and reactions

This individual has the following underlying tendencies:

{constants.NEUROTYPE_TRAIT_PARAGRAPHS.get(neurotype)}

Trait alignment requirement:
- Every memory must be behaviorally consistent with the group's tendencies above.
- Avoid writing memories that primarily reflect the opposite group style.

IMPORTANT:
- Do NOT explicitly mention or restate these traits
- Do NOT describe the person using labels
- Instead, let these traits influence:
  - what situations stand out
  - how events are interpreted
  - how the individual reflects on experiences

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

Guidelines:
- Avoid repetitive phrasing or patterns
- Avoid overly formal or robotic language
- Do NOT explain behavior using abstract traits
- Show behavior through concrete experiences
- Reflections should feel natural, not analytical essays

Use these seed prompts for inspiration (do not copy verbatim):
{prompt_block}

Output rules:
- Return exactly {remaining} memories
- One memory per line
- Do not number lines
- Do not include commentary or headings

Already generated memories in prior attempts (avoid repeating):
{chr(10).join(f'- {m}' for m in out[-10:]) if out else '- (none)'}

"""

      raw = self._sample_text_with_retries(generation_prompt)
      lines = [line.strip() for line in (raw or '').splitlines() if line.strip()]
      for line in lines:
        clean = self._normalize_memory_line(line=line, agent_name=agent_name)
        if not clean:
          continue
        memory_text = f"{agent_name} remembers: {clean}"
        if memory_text in seen:
          continue
        seen.add(memory_text)
        out.append(memory_text)
        if len(out) >= count:
          break

    if len(out) < count:
      raise RuntimeError(
          f'Could not generate enough unique formative memories for {agent_name}. '
          f'Requested {count}, generated {len(out)} after {max_attempts} attempts.'
      )

    return out[:count]

  def _sample_text_with_retries(self, prompt: str) -> str:
    """Retry transient model API failures with exponential backoff."""
    max_attempts = 5
    base_delay_seconds = 1.0
    for attempt in range(1, max_attempts + 1):
      try:
        return self._model.sample_text(prompt)
      except Exception as exc:
        message = str(exc).lower()
        # Treat common upstream transient failures as retryable.
        is_retryable = (
            'internal error encountered' in message
            or 'internalservererror' in message
            or 'statuscode.internal' in message
            or 'service unavailable' in message
            or 'unavailable' in message
            or 'deadline exceeded' in message
            or 'resource exhausted' in message
            or 'too many requests' in message
            or '429' in message
            or '500' in message
            or '503' in message
        )

        if not is_retryable or attempt == max_attempts:
          raise

        # Exponential backoff with small jitter helps avoid synchronized retries.
        delay = base_delay_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.5)
        print(
            f'Transient model error during formative memory generation '
            f'(attempt {attempt}/{max_attempts}): {exc}. Retrying in {delay:.1f}s...'
        )
        time.sleep(delay)

  def _normalize_memory_line(self, *, line: str, agent_name: str) -> str:
    """Normalize a generated memory line and strip duplicated name prefixes."""
    clean = line.lstrip('0123456789.-) ').strip()
    if not clean:
      return ''

    prefix_pattern = re.compile(
        rf'^(?:{re.escape(agent_name)}\s+remembers:\s*)+',
        flags=re.IGNORECASE,
    )
    clean = prefix_pattern.sub('', clean).strip()
    clean = clean.strip('"\' ')
    return clean

  def initialize_entity_memories(self, entity, memories: list[str]) -> None:
    """Inject memories into both standard memory and IMPE memory components."""
    std_memory = entity.get_component(
        agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY,
        type_=agent_components.memory.AssociativeMemory,
    )
    impe_memory = entity.get_component(
        impe_components.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe_components.IMPEMemoryComponent,
    )

    for idx, memory in enumerate(memories, start=1):
      std_memory.add(memory)
      impe_memory.add_observation(
          turn=0,
          observed_from='formative_memory',
          text=f'[m{idx}] {memory}',
          body='',
      )

  def initialize_candidate_and_interviewer(
      self,
      *,
      candidate,
      interviewer,
      candidate_neurotype: str,
      interviewer_neurotype: str,
      memory_count: int = 24,
  ) -> tuple[list[str], list[str]]:
    """Generate and inject formative memories for both entities."""
    candidate_memories = self.generate_memories(
        agent_name=candidate.name,
        role='candidate',
        neurotype=candidate_neurotype,
        prompts=constants.MEMORY_PROMPTS,
        count=memory_count,
    )
    interviewer_memories = self.generate_memories(
        agent_name=interviewer.name,
        role='interviewer',
        neurotype=interviewer_neurotype,
        prompts=constants.MEMORY_PROMPTS,
        count=memory_count,
    )
    self.initialize_entity_memories(candidate, candidate_memories)
    self.initialize_entity_memories(interviewer, interviewer_memories)
    return candidate_memories, interviewer_memories

  # def _fallback_memories(self, agent_name: str, missing: int) -> list[str]:
  #   base = [
  #     f'{agent_name} remembers: A direct communication choice improved an outcome.',
  #     f'{agent_name} remembers: A misunderstanding happened when assumptions were left implicit.',
  #     f'{agent_name} remembers: Clarifying expectations prevented a service error.',
  #     f'{agent_name} remembers: Slower pacing made a difficult conversation manageable.',
  #   ]
  #   out: list[str] = []
  #   while len(out) < missing:
  #     out.extend(base)
  #   return out[:missing]
