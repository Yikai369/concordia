"""Utility functions for Impression Management PE Conversation."""

import datetime
import os
import random
import re
from typing import Any

from concordia.components.agent.impression_management_pe import (
    CulturalNorm,
    PersonalityTrait,
)


def parse_index_list(s: str | None) -> list[int]:
  """Parse comma-separated 1-based indices into 0-based list; ignore empties."""
  if not s:
    return []
  parts = [p.strip() for p in s.split(',') if p.strip() != '']
  out = []
  for p in parts:
    try:
      i = int(p)
      if i >= 1:
        out.append(i - 1)
    except Exception:
      continue
  return out


def select_by_indices(full_list: list[Any], indices: list[int]) -> list[Any]:
  """Select items from full_list by indices."""
  return [full_list[i] for i in indices if 0 <= i < len(full_list)]


def generate_trait_scores(
    rng: random.Random,
    trait_list: list[PersonalityTrait],
    is_audience: bool,
) -> dict[str, int]:
  """Generate trait scores (audience: 2-3, actor: 0-1)."""
  scores: dict[str, int] = {}
  for t in trait_list:
    if is_audience:
      scores[t.name] = rng.randint(2, 3)
    else:
      scores[t.name] = rng.randint(0, 1)
  return scores


def create_output_directory(save_dir: str | None) -> str:
  """Create timestamped directory if save_dir is None."""
  if save_dir is None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join('./temp', timestamp)
  os.makedirs(save_dir, exist_ok=True)
  return save_dir


def format_conversation(utterances: list) -> str:
  """Format conversation for prompts."""
  if not utterances:
    return '- (none)'
  return '\n'.join(
      f'- [t={u.turn} {u.speaker}] {u.text}' for u in utterances
  )


def parse_dialogue_and_body(response: str) -> tuple[str, str]:
  """Parse dialogue and body language from response."""
  m1 = re.search(r'DIALOGUE:\s*(.*)', response)
  m2 = re.search(r'BODY:\s*(.*)', response)
  dialogue = m1.group(1).strip() if m1 else response.strip()
  body = m2.group(1).strip() if m2 else ''
  return dialogue, body


def extract_numeric_from_response(
    response: str, default: float = 0.5
) -> float:
  """Extract numeric value from LLM response."""
  m = re.search(r'([01](?:\.\d+)?)', response)
  value = float(m.group(1)) if m else default
  return clamp_to_range(value)


def clamp_to_range(
    value: float, min_val: float = 0.0, max_val: float = 1.0
) -> float:
  """Clamp value to [min_val, max_val]."""
  return max(min_val, min(max_val, value))
