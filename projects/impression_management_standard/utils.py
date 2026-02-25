"""Utility functions for Impression Management PE Conversation."""

import csv
import datetime
import os
import random
import re
from typing import Any

from concordia.components.agent.impression_management_pe import (
    CulturalNorm,
    PersonalityTrait,
)


def load_traits_from_spreadsheet(file_path: str) -> list[PersonalityTrait]:
    """Load personality traits from Excel (.xlsx) or CSV file.

    Expects columns 'name' and 'assertion' (case-insensitive). For CSV, the
    first row is the header. For Excel, the first row is the header.

    Args:
        file_path: Path to .xlsx or .csv file.

    Returns:
        List of PersonalityTrait (name, assertion). Empty list if sheet has
        no data rows.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If required columns are missing or file format is unsupported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Traits file not found: {file_path}")

    path_lower = file_path.lower()
    if path_lower.endswith('.csv'):
        return _load_traits_from_csv(file_path)
    if path_lower.endswith('.xlsx') or path_lower.endswith('.xls'):
        return _load_traits_from_excel(file_path)
    # Default: try CSV first, then suggest format
    try:
        return _load_traits_from_csv(file_path)
    except Exception:
        raise ValueError(
            f"Unsupported traits file format: {file_path}. Use .csv or .xlsx (Excel). "
            "For Excel, install: pip install openpyxl"
        )


def _load_traits_from_csv(file_path: str) -> list[PersonalityTrait]:
    """Load traits from CSV. Header: name, assertion (case-insensitive)."""
    traits: list[PersonalityTrait] = []
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        fieldnames_lower = [s.strip().lower() for s in fieldnames]
        if 'name' not in fieldnames_lower or 'assertion' not in fieldnames_lower:
            raise ValueError(
                f"CSV must have columns 'name' and 'assertion'. Found: {fieldnames}"
            )
        name_key = fieldnames[fieldnames_lower.index('name')]
        assertion_key = fieldnames[fieldnames_lower.index('assertion')]
        for row in reader:
            name = (row.get(name_key) or '').strip()
            assertion = (row.get(assertion_key) or '').strip()
            if name or assertion:
                traits.append(PersonalityTrait(name=name or 'Unnamed', assertion=assertion or ''))
    return traits


def _load_traits_from_excel(file_path: str) -> list[PersonalityTrait]:
    """Load traits from Excel (.xlsx). First row = header with 'name' and 'assertion'."""
    try:
        import openpyxl
    except ImportError as e:
        raise ValueError(
            "Loading Excel (.xlsx) requires openpyxl. Install with: pip install openpyxl"
        ) from e
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    if ws is None:
        wb.close()
        return []
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        return []
    header = [str(c).strip().lower() if c is not None else '' for c in rows[0]]
    if 'name' not in header or 'assertion' not in header:
        raise ValueError(
            f"Excel sheet must have columns 'name' and 'assertion'. Found: {rows[0]}"
        )
    name_col = header.index('name')
    assertion_col = header.index('assertion')
    traits: list[PersonalityTrait] = []
    for row in rows[1:]:
        if len(row) <= max(name_col, assertion_col):
            continue
        name = str(row[name_col] or '').strip()
        assertion = str(row[assertion_col] or '').strip()
        if name or assertion:
            traits.append(PersonalityTrait(name=name or 'Unnamed', assertion=assertion or ''))
    return traits


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




