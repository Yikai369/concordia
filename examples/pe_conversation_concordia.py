#!/usr/bin/env python3
"""
PE Conversation in Concordia Framework
-------------------------------------
Two LLM agents conversing with PE-driven adaptation using Concordia.

This is a Concordia-based implementation of the PE conversation system.
"""

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import datetime
import json
import os
import sys
from typing import Any

from concordia.components.agent import pe_conversation as pe_components
from concordia.language_model import utils as language_model_utils
from concordia.prefabs.entity import pe_entity
import concordia.prefabs.entity as entity_prefabs
from concordia.prefabs.game_master import generic as game_master_prefabs
import concordia.prefabs.game_master as gm_prefabs
from concordia.prefabs.simulation import generic as simulation
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
import numpy as np
import sentence_transformers


@dataclass
class TurnLog:
  """Log entry for a single turn."""
  time: str
  turn: int
  speaker: str
  listener: str
  speaker_text: str
  listener_estimate: float
  listener_pe: float
  listener_reflection: str


def extract_turn_data_from_log(
    raw_log: list[dict[str, Any]],
    agent_a_name: str,
    agent_b_name: str,
) -> list[TurnLog]:
  """Extract turn data from Concordia raw log.

  Args:
    raw_log: Raw log from simulation.
    agent_a_name: Name of agent A.
    agent_b_name: Name of agent B.

  Returns:
    List of TurnLog entries.
  """
  turn_logs = []
  current_turn = 0

  for step in raw_log:
    step_num = step.get('Step', 0)
    summary = step.get('Summary', '')

    # Look for entity actions and PE data
    for key, value in step.items():
      if isinstance(value, dict):
        # Check if this is an entity entry
        if agent_a_name in key or agent_b_name in key:
          # Try to extract action text
          action_text = None
          if 'ActComponent' in value:
            action_text = value['ActComponent'].get('Value', '')
          elif 'Value' in value:
            action_text = value['Value']

          # Determine speaker and listener
          if agent_a_name in key:
            speaker = agent_a_name
            listener = agent_b_name
          else:
            speaker = agent_b_name
            listener = agent_a_name

          # Extract PE data from listener's components
          listener_estimate = 0.0
          listener_pe = 0.0
          listener_reflection = ''

          # Look for listener's PE components in the log
          listener_key = None
          for k in step.keys():
            if listener in k:
              listener_key = k
              break

          if listener_key and listener_key in step:
            listener_data = step[listener_key]
            if isinstance(listener_data, dict):
              # Extract PE estimation
              if 'PE_Estimation' in listener_data:
                est_data = listener_data['PE_Estimation'].get('Value', '')
                # Parse estimate and PE from string like "Estimated state: 0.75, PE: +0.25"
                import re
                est_match = re.search(r'estimate[:\s]+([\d.]+)', est_data, re.I)
                pe_match = re.search(r'PE[:\s]+([+-]?[\d.]+)', est_data, re.I)
                if est_match:
                  listener_estimate = float(est_match.group(1))
                if pe_match:
                  listener_pe = float(pe_match.group(1))

              # Extract reflection
              if 'PE_Reflection' in listener_data:
                listener_reflection = listener_data['PE_Reflection'].get('Value', '')

          if action_text:
            current_turn += 1
            turn_logs.append(
                TurnLog(
                    time=datetime.datetime.utcnow().isoformat() + 'Z',
                    turn=current_turn,
                    speaker=speaker,
                    listener=listener,
                    speaker_text=action_text,
                    listener_estimate=listener_estimate,
                    listener_pe=listener_pe,
                    listener_reflection=listener_reflection,
                )
            )

  return turn_logs


def extract_turn_data_from_entities(
    sim: simulation.Simulation,
    agent_a_name: str,
    agent_b_name: str,
    total_turns: int,
) -> list[TurnLog]:
  """Extract turn data directly from entity components.

  Args:
    sim: The simulation object.
    agent_a_name: Name of agent A.
    agent_b_name: Name of agent B.
    total_turns: Total number of turns.

  Returns:
    List of TurnLog entries.
  """
  turn_logs = []

  # Get entities
  entities = {e.name: e for e in sim.get_entities()}
  agent_a = entities.get(agent_a_name)
  agent_b = entities.get(agent_b_name)

  if not agent_a or not agent_b:
    return turn_logs

  # Access PE memory components
  from concordia.components.agent import pe_conversation as pe_comp

  def get_pe_memory(entity):
    if hasattr(entity, 'get_component'):
      return entity.get_component(
          pe_comp.DEFAULT_PE_MEMORY_COMPONENT_KEY,
          type_=pe_comp.PEMemoryComponent,
      )
    return None

  mem_a = get_pe_memory(agent_a)
  mem_b = get_pe_memory(agent_b)

  if not mem_a or not mem_b:
    return turn_logs

  # Extract data from memory
  conv_a = mem_a.get_recent_conversation(k=total_turns)
  conv_b = mem_b.get_recent_conversation(k=total_turns)
  pe_a = mem_b.get_recent_pe_history(k=total_turns)  # A's PE from B's perspective
  pe_b = mem_a.get_recent_pe_history(k=total_turns)  # B's PE from A's perspective
  refl_a = mem_b.get_recent_reflections(k=total_turns)
  refl_b = mem_a.get_recent_reflections(k=total_turns)

  # Combine and sort by turn
  all_utterances = conv_a + conv_b
  all_utterances.sort(key=lambda x: x.turn)

  for i, utterance in enumerate(all_utterances):
    if utterance.speaker == agent_a_name:
      speaker = agent_a_name
      listener = agent_b_name
      # Find corresponding PE and reflection from B's perspective
      pe_rec = next((p for p in pe_b if p.turn == utterance.turn), None)
      refl = next((r for r in refl_b if r.turn == utterance.turn), None)
    else:
      speaker = agent_b_name
      listener = agent_a_name
      # Find corresponding PE and reflection from A's perspective
      pe_rec = next((p for p in pe_a if p.turn == utterance.turn), None)
      refl = next((r for r in refl_a if r.turn == utterance.turn), None)

    if pe_rec:
      turn_logs.append(
          TurnLog(
              time=datetime.datetime.utcnow().isoformat() + 'Z',
              turn=utterance.turn,
              speaker=speaker,
              listener=listener,
              speaker_text=utterance.text,
              listener_estimate=pe_rec.estimate,
              listener_pe=pe_rec.pe,
              listener_reflection=refl.text if refl else '',
          )
      )

  return turn_logs


def main():
  parser = argparse.ArgumentParser(
      description='Two-agent PE conversation (Concordia framework).'
  )
  parser.add_argument(
      '--turns',
      type=int,
      default=6,
      help='Total turns (messages) in the dialogue.',
  )
  parser.add_argument(
      '--model',
      type=str,
      default='gpt-4o-mini',
      help='Model name (OpenAI or Together AI).',
  )
  parser.add_argument(
      '--api_type',
      type=str,
      default='openai',
      help='API type: openai or together_ai',
  )
  parser.add_argument(
      '--temperature',
      type=float,
      default=0.2,
      help='Sampling temperature.',
  )
  parser.add_argument(
      '--top_p',
      type=float,
      default=0.9,
      help='Top-p nucleus sampling.',
  )
  parser.add_argument(
      '--window',
      type=int,
      default=3,
      help='Recent K turns to condition on.',
  )
  parser.add_argument(
      '--outfile',
      type=str,
      default='pe_conversation_log.json',
      help='Where to save JSON log.',
  )
  parser.add_argument(
      '--api_key',
      type=str,
      default='',
      help='API key (or set OPENAI_API_KEY env var).',
  )
  args = parser.parse_args()

  # Get API key
  api_key = args.api_key or os.environ.get('OPENAI_API_KEY', '')
  if not api_key:
    print(
        'ERROR: API key required. Set OPENAI_API_KEY env var or use --api_key',
        file=sys.stderr,
    )
    sys.exit(1)

  # Setup language model
  model = language_model_utils.language_model_setup(
      api_type=args.api_type,
      model_name=args.model,
      api_key=api_key,
      disable_language_model=False,
  )

  # Setup embedder
  st_model = sentence_transformers.SentenceTransformer(
      'sentence-transformers/all-mpnet-base-v2'
  )
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)

  # Load prefabs
  prefabs = {
      **helper_functions.get_package_classes(entity_prefabs),
      **helper_functions.get_package_classes(gm_prefabs),
  }

  # Define goal
  goal_name = 'likability'
  goal_description = (
      'Be perceived as likable by the partner '
      '(0=not liked, 1=fully liked). Aim for 1.0.'
  )
  goal_ideal = 1.0

  # Create instances
  agent_a_name = 'Agent A'
  agent_b_name = 'Agent B'

  instances = [
      prefab_lib.InstanceConfig(
          prefab='pe_entity__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': agent_a_name,
              'goal_name': goal_name,
              'goal_description': goal_description,
              'goal_ideal': goal_ideal,
              'recent_k': args.window,
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='pe_entity__Entity',
          role=prefab_lib.Role.ENTITY,
          params={
              'name': agent_b_name,
              'goal_name': goal_name,
              'goal_description': goal_description,
              'goal_ideal': goal_ideal,
              'recent_k': args.window,
          },
      ),
      prefab_lib.InstanceConfig(
          prefab='dialogic__GameMaster',
          role=prefab_lib.Role.GAME_MASTER,
          params={
              'name': 'PE Conversation GM',
              'acting_order': 'fixed',  # Alternate between agents
          },
      ),
  ]

  # Create config
  config = prefab_lib.Config(
      default_premise=(
          'Two agents are having a conversation. '
          'Each agent adapts their responses based on prediction error.'
      ),
      default_max_steps=args.turns * 2,  # Each turn = act + observe
      prefabs=prefabs,
      instances=instances,
  )

  # Initialize simulation
  sim = simulation.Simulation(
      config=config,
      model=model,
      embedder=embedder,
  )

  # Run simulation
  raw_log = []
  results_log = sim.play(max_steps=args.turns * 2, raw_log=raw_log)

  # Extract turn data
  turn_logs = extract_turn_data_from_entities(
      sim, agent_a_name, agent_b_name, args.turns
  )

  # If extraction from entities failed, try from log
  if not turn_logs:
    turn_logs = extract_turn_data_from_log(
        raw_log, agent_a_name, agent_b_name
    )

  # Pretty print
  for r in turn_logs:
    print(f'[t={r.turn}] {r.speaker} -> {r.listener}: {r.speaker_text}')
    print(
        f'       {r.listener} observed estimate={r.listener_estimate:.2f}, '
        f'PE={r.listener_pe:+.2f}'
    )
    print(f'       {r.listener} reflection: {r.listener_reflection}\n')

  # Save JSON
  with open(args.outfile, 'w', encoding='utf-8') as f:
    json.dump(
        [asdict(l) for l in turn_logs],
        f,
        ensure_ascii=False,
        indent=2,
    )
  print(f'Saved detailed log -> {args.outfile}')


if __name__ == '__main__':
  main()
