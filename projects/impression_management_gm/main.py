#!/usr/bin/env python3
"""Entry point for the impression_management_gm experiment."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import random
import sys


def _bootstrap_repo_root() -> None:
  """Ensure repository root is importable when run via python -m main."""
  this_file = Path(__file__).resolve()
  # .../concordia/projects/impression_management_gm/main.py -> .../concordia
  repo_root = this_file.parents[2]
  repo_root_str = str(repo_root)
  if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)


_bootstrap_repo_root()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description='Run game-master based impression management experiment.'
  )
  parser.add_argument(
      '--model',
      default='gemini-flash-latest',
    help=(
      'Model name for the selected backend. Defaults to a Gemini model '
      'unless --local_model is enabled.'
    ),
  )
  parser.add_argument(
    '--local_model',
    action='store_true',
    help='Use NVIDIA OpenAI-compatible endpoint instead of Gemini.',
  )
  parser.add_argument(
    '--local_model_name',
    type=str,
    default='nvidia/nemotron-3-super-120b-a12b',
    help='Model name to use when --local_model is enabled.',
  )
  parser.add_argument(
    '--local_api_base',
    type=str,
    default='https://integrate.api.nvidia.com/v1',
    help='Base URL for the local OpenAI-compatible endpoint.',
  )
  parser.add_argument(
      '--condition',
      default='all',
      choices=[
          'all',
          'Riffer_x_Riffer',
          'Caden_x_Caden',
          'Riffer_x_Caden',
          'Caden_x_Riffer',
      ],
      help='[Candidate neurotype] x [interviewer neurotype]'
  )
  parser.add_argument('--turns', type=int, default=6)
  parser.add_argument('--memory_count', type=int, default=24)
  parser.add_argument('--save_dir', type=str, default=None)
  parser.add_argument(
      '--interview',
      action='store_true',
      help='Run the interview loop instead of the questionnaire. Default is to run questionnaire only.',
  )
  return parser.parse_args()


def _condition_pairs(
    label: str,
    experiment_conditions: list[tuple[str, str]],
) -> list[tuple[str, str]]:
  if label == 'all':
    return list(experiment_conditions)
  left, right = label.split('_x_')
  return [(left, right)]


def _avg(values: list[float]) -> float:
  return sum(values) / len(values) if values else 0.0


def _sample_agent_names(*, rng: random.Random, pool: list[str]) -> tuple[str, str]:
  """Sample distinct candidate/interviewer names from a shared pool."""
  if len(pool) < 2:
    raise ValueError('AGENT_NAME_POOL must include at least two names.')
  candidate_name, interviewer_name = rng.sample(pool, 2)
  return candidate_name, interviewer_name


def _write_json(path: Path, payload: object) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)


def main() -> None:
  args = parse_args()

  from concordia.language_model import google_aistudio_model
  from concordia.language_model import nvidia_openai_model

  from projects.impression_management import setup as base_setup
  from projects.impression_management_gm import constants
  from projects.impression_management_gm import entities
  from projects.impression_management_gm import formative_memories_initializer
  from projects.impression_management_gm import questionnaire
  from projects.impression_management_gm.game_master import InterviewGameMaster

  if args.local_model:
    api_key = os.environ.get('NVIDIA_API_KEY', '').strip()
    if not api_key:
      raise RuntimeError(
          'Set NVIDIA_API_KEY when --local_model is enabled.'
      )
    model = nvidia_openai_model.NvidiaOpenAILanguageModel(
        model_name=args.local_model_name,
        api_key=api_key,
        api_base=args.local_api_base,
    )
  else:
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
      # GoogleAIStudioLanguageModel defaults to GOOGLE_API_KEY; keep both for convenience.
      api_key = os.environ.get('GOOGLE_API_KEY', '').strip()
    if not api_key:
      raise RuntimeError(
          'Set GOOGLE_API_KEY for Gemini access.'
      )

    model = google_aistudio_model.GoogleAIStudioLanguageModel(
        model_name=args.model,
        api_key=api_key,
    )

  save_dir = Path(args.save_dir or f"./temp/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
  save_dir.mkdir(parents=True, exist_ok=True)

  _, memory_bank = base_setup.setup_embedder_and_memory()

  initializer = formative_memories_initializer.FormativeMemoriesInitializer(model)
  rng = random.Random(constants.DEFAULT_SEED)

  if args.interview:
    selected_conditions = _condition_pairs(
        args.condition,
        constants.EXPERIMENT_CONDITIONS,
    )
  else:
    if args.condition == 'all':
      selected_conditions = [('Riffer', 'Caden')]
    else:
      selected_conditions = _condition_pairs(
          args.condition,
          constants.EXPERIMENT_CONDITIONS,
      )

  aggregate: list[dict] = []
  questionnaire_results: list[dict] = []

  for candidate_neurotype, interviewer_neurotype in selected_conditions:
    candidate_name, interviewer_name = _sample_agent_names(
        rng=rng,
        pool=constants.AGENT_NAME_POOL,
    )

    condition_id = f'{candidate_neurotype}_x_{interviewer_neurotype}'
    condition_dir = save_dir / condition_id
    condition_dir.mkdir(parents=True, exist_ok=True)

    candidate, interviewer = entities.build_candidate_and_interviewer(
        model=model,
        memory_bank=memory_bank,
      candidate_name=candidate_name,
      interviewer_name=interviewer_name,
        candidate_neurotype=candidate_neurotype,
        interviewer_neurotype=interviewer_neurotype,
    )

    candidate_memories, interviewer_memories = (
        initializer.initialize_candidate_and_interviewer(
            candidate=candidate,
            interviewer=interviewer,
            candidate_neurotype=candidate_neurotype,
            interviewer_neurotype=interviewer_neurotype,
            memory_count=max(20, min(50, args.memory_count)),
        )
    )

    _write_json(
      condition_dir / 'memories.json',
      {
        'candidate_memories': candidate_memories,
        'interviewer_memories': interviewer_memories,
      },
    )

    if not args.interview:
      qr_candidate = {
        'convergent_validity': questionnaire.run_convergent_validity_questionnaire(
          model=model,
          agent_name=candidate.name,
          role_label='candidate',
          neurotype=candidate_neurotype,
          role_context=constants.CANDIDATE_ROLE_CONTEXT,
          memories=candidate_memories,
        ),
        'empathy': questionnaire.run_empathy_questionnaire(
          model=model,
          agent_name=candidate.name,
          role_label='candidate',
          neurotype=candidate_neurotype,
          role_context=constants.CANDIDATE_ROLE_CONTEXT,
          memories=candidate_memories,
        ),
      }
      qr_interviewer = {
        'convergent_validity': questionnaire.run_convergent_validity_questionnaire(
          model=model,
          agent_name=interviewer.name,
          role_label='interviewer',
          neurotype=interviewer_neurotype,
          role_context=constants.INTERVIEWER_ROLE_CONTEXT,
          memories=interviewer_memories,
        ),
        'empathy': questionnaire.run_empathy_questionnaire(
          model=model,
          agent_name=interviewer.name,
          role_label='interviewer',
          neurotype=interviewer_neurotype,
          role_context=constants.INTERVIEWER_ROLE_CONTEXT,
          memories=interviewer_memories,
        ),
      }
      questionnaire_result = {
        'condition': condition_id,
        'candidate': qr_candidate,
        'interviewer': qr_interviewer,
      }
      questionnaire_results.append(questionnaire_result)
      _write_json(condition_dir / 'questionnaire.json', questionnaire_result)
      continue

    gm = InterviewGameMaster(
        model=model,
        candidate=candidate,
        interviewer=interviewer,
        candidate_neurotype=candidate_neurotype,
        interviewer_neurotype=interviewer_neurotype,
    )

    rows = gm.run(turns=args.turns)
    gm.save_json(str(condition_dir / 'interaction_log.json'))

    aggregate.append(
        {
            'condition': condition_id,
            'candidate_neurotype': candidate_neurotype,
            'interviewer_neurotype': interviewer_neurotype,
            'turns': len(rows),
            'candidate_competence_avg': _avg([r.candidate_competence for r in rows]),
        }
    )

  if not args.interview:
    _write_json(
        save_dir / 'questionnaire_results.json',
        {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'results': questionnaire_results,
        },
    )
  else:
    with open(save_dir / 'aggregate_results.json', 'w', encoding='utf-8') as f:
      json.dump(
          {
              'generated_at': datetime.utcnow().isoformat() + 'Z',
              'results': aggregate,
          },
          f,
          indent=2,
      )

  print(f'Completed. Results saved in: {save_dir}')


if __name__ == '__main__':
  main()
