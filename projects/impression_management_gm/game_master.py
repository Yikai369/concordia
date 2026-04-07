"""Custom game master loop for neurotype-based IMPE interview experiments."""

from dataclasses import asdict
from dataclasses import dataclass
import json
import re
from typing import Any

from concordia.components.agent import \
    impression_management_pe as impe_components


@dataclass
class TurnMetrics:
  turn: int
  candidate_neurotype: str
  interviewer_neurotype: str
  interviewer_message: str
  candidate_response: str
  candidate_competence: float


@dataclass
class MessageLogEntry:
  sequence: int
  turn: int
  speaker: str
  role: str
  message: str
  body: str


class InterviewGameMaster:
  """Runs interview turns using IMPE entities and computes per-turn metrics."""

  def __init__(
      self,
      *,
      model,
      candidate,
      interviewer,
      candidate_neurotype: str,
      interviewer_neurotype: str,
  ):
    self._model = model
    self._candidate = candidate
    self._interviewer = interviewer
    self._candidate_neurotype = candidate_neurotype
    self._interviewer_neurotype = interviewer_neurotype
    self._log: list[TurnMetrics] = []
    self._messages: list[MessageLogEntry] = []
    self._candidate_assessments_by_turn: dict[int, dict[str, Any] | None] = {}
    self._interviewer_assessments_by_turn: dict[int, dict[str, Any] | None] = {}
    self._prior_impression_by_turn: dict[int, dict[str, Any] | None] = {}
    self._posterior_impression_by_turn: dict[int, dict[str, Any] | None] = {}
    self._feedback_interpretation_by_turn: dict[int, dict[str, Any] | None] = {}

  def run(self, turns: int) -> list[TurnMetrics]:
    """Run turn-structured interaction.

    Turn 0: interviewer greeting.
    Turn t>=1: candidate responds, interviewer responds.
    """
    candidate_memory, interviewer_memory, audience_eval, actor_pf, reflection = (
        self._conversation_components()
    )

    interviewer_message = self._interviewer_greet(turn=0)
    self._append_message(
      turn=0,
      speaker=self._interviewer.name,
      role='interviewer',
      message=interviewer_message,
      body='',
    )
    interviewer_memory.add_utterance(
        turn=0,
        actor=self._interviewer.name,
        text=interviewer_message,
        body='',
    )
    candidate_memory.add_observation(
        turn=0,
        observed_from=self._interviewer.name,
        text=interviewer_message,
        body='',
    )

    for turn in range(1, turns + 1):
      candidate_action = self._candidate.act()
      candidate_text, candidate_body = self._parse_dialogue_and_body(candidate_action)

      prior_impression = self._candidate.get_component(
          impe_components.DEFAULT_PRIOR_IMPRESSION_COMPONENT_KEY,
          type_=impe_components.PriorImpressionComponent,
      )
      if prior_impression:
        state = prior_impression.get_state()
        self._prior_impression_by_turn[turn] = {
            'prior_impression_score': state.get('last_score', 'D'),
            'prior_impression_confidence': state.get('last_confidence', 'D'),
        }
      else:
        self._prior_impression_by_turn[turn] = None

      self._candidate_assessments_by_turn[turn] = self._extract_self_assessment(
          self._candidate
      )

      observation = (
          f'{self._candidate.name} said: "{candidate_text}"\n'
          f'Body language: "{candidate_body}"'
      )
      audience_eval.pre_observe(observation)
      audience_eval.post_observe()

      evaluations = interviewer_memory.get_recent_evaluations()
      if evaluations:
        latest = evaluations[-1].utterance
        actor_pf.pre_observe(
            f'{latest.actor} said: "{latest.text}"\nBody language: "{latest.body}"'
        )
        actor_pf.post_observe()
      reflection.post_observe()

      interviewer_message, interviewer_body = self._interviewer_follow_up(turn)
      self._interviewer_assessments_by_turn[turn] = self._extract_self_assessment(
          self._interviewer
        )

      self._append_message(
          turn=turn,
          speaker=self._candidate.name,
          role='candidate',
          message=candidate_text,
          body=candidate_body,
      )

      self._append_message(
          turn=turn,
          speaker=self._interviewer.name,
          role='interviewer',
          message=interviewer_message,
          body=interviewer_body,
      )

      candidate_memory.add_action(turn=turn, text=candidate_text, body=candidate_body)
      candidate_memory.add_utterance(
          turn=turn,
          actor=self._candidate.name,
          text=candidate_text,
          body=candidate_body,
      )
      interviewer_memory.add_observation(
          turn=turn,
          observed_from=self._candidate.name,
          text=candidate_text,
          body=candidate_body,
      )
      interviewer_memory.add_utterance(
          turn=turn,
          actor=self._interviewer.name,
          text=interviewer_message,
          body=interviewer_body,
      )
      candidate_memory.add_observation(
          turn=turn,
          observed_from=self._interviewer.name,
          text=interviewer_message,
          body=interviewer_body,
      )

      # Posterior impression and feedback interpretation components
      # Run these after interviewer responds so candidate can react to feedback
      posterior_impression = self._candidate.get_component(
          impe_components.DEFAULT_POSTERIOR_IMPRESSION_COMPONENT_KEY,
          type_=impe_components.PosteriorImpressionComponent,
      )
      feedback_interpretation = self._candidate.get_component(
          impe_components.DEFAULT_FEEDBACK_INTERPRETATION_COMPONENT_KEY,
          type_=impe_components.FeedbackInterpretationComponent,
      )
      if posterior_impression:
        posterior_impression.post_observe()
        state = posterior_impression.get_state()
        self._posterior_impression_by_turn[turn] = {
            'posterior_impression_score': state.get('last_score', 'D'),
            'posterior_impression_confidence': state.get('last_confidence', 'D'),
        }
      else:
        self._posterior_impression_by_turn[turn] = None

      if feedback_interpretation:
        feedback_interpretation.post_observe()
        state = feedback_interpretation.get_state()
        self._feedback_interpretation_by_turn[turn] = {
            'feedback_interpretation': state.get('last_interpretation', 'C'),
            'interpretation_confidence': state.get('last_confidence', 'D'),
            'prediction_error': state.get('last_surprise', 'C'),
        }
      else:
        self._feedback_interpretation_by_turn[turn] = None

      metrics = TurnMetrics(
          turn=turn,
          candidate_neurotype=self._candidate_neurotype,
          interviewer_neurotype=self._interviewer_neurotype,
          interviewer_message=interviewer_message,
          candidate_response=candidate_text,
          candidate_competence=self._score_competence(candidate_text),
      )
      self._log.append(metrics)

    return self._log

  def save_json(self, output_path: str) -> None:
    metrics_by_turn = {row.turn: row for row in self._log}
    turns = sorted({message.turn for message in self._messages})
    turn_rows: list[dict] = []

    for turn in turns:
      messages = [
          asdict(message)
          for message in self._messages
          if message.turn == turn
      ]
      metrics = metrics_by_turn.get(turn)
      if metrics is None:
        scores = None
      else:
        scores = {
            'candidate_competence': metrics.candidate_competence,
            'candidate_self_assessment': self._candidate_assessments_by_turn.get(turn),
            'interviewer_self_assessment': self._interviewer_assessments_by_turn.get(turn),
            'prior_impression': self._prior_impression_by_turn.get(turn),
            'posterior_impression': self._posterior_impression_by_turn.get(turn),
            'feedback_interpretation': self._feedback_interpretation_by_turn.get(turn),
        }

      turn_rows.append(
          {
              'turn': turn,
              'messages': messages,
              'scores': scores,
          }
      )

    with open(output_path, 'w', encoding='utf-8') as f:
      json.dump(turn_rows, f, indent=2)

  def _append_message(
      self,
      *,
      turn: int,
      speaker: str,
      role: str,
      message: str,
      body: str,
  ) -> None:
    self._messages.append(
        MessageLogEntry(
            sequence=len(self._messages),
            turn=turn,
            speaker=speaker,
            role=role,
            message=message,
            body=body,
        )
    )

  def _extract_self_assessment(self, agent: Any) -> dict[str, Any] | None:
    if not hasattr(agent, 'get_last_log'):
      return None
    last_log = agent.get_last_log()
    if not isinstance(last_log, dict):
      return None
    act_log = last_log.get('__act__')
    if not isinstance(act_log, dict):
      return None
    if act_log.get('Key') != 'Self-Assessment':
      return None
    return {
        'is_acceptable': act_log.get('Is Acceptable'),
        'was_revised': act_log.get('Was Revised'),
        'feedback': act_log.get('Feedback'),
        'posthoc_reasoning': act_log.get('Posthoc Reasoning'),
        'original_response': act_log.get('Original Response'),
        'final_response': act_log.get('Final Response'),
      'option_space': act_log.get('Option Space'),
    }

  def _conversation_components(self):
    candidate_memory = self._candidate.get_component(
        impe_components.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe_components.IMPEMemoryComponent,
    )
    interviewer_memory = self._interviewer.get_component(
        impe_components.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe_components.IMPEMemoryComponent,
    )
    audience_eval = self._interviewer.get_component(
        impe_components.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY,
        type_=impe_components.IMPEAudienceEvaluationComponent,
    )
    actor_pf = self._candidate.get_component(
        impe_components.DEFAULT_IMPE_ACTOR_PARTICLE_FILTER_COMPONENT_KEY,
        type_=impe_components.IMPEActorParticleFilterComponent,
    )
    reflection = self._candidate.get_component(
        impe_components.DEFAULT_IMPE_REFLECTION_COMPONENT_KEY,
        type_=impe_components.IMPEReflectionComponent,
    )
    return candidate_memory, interviewer_memory, audience_eval, actor_pf, reflection

  def _interviewer_greet(self, turn: int) -> str:
    del turn
    greeting = self._model.sample_text(
        f'You are {self._interviewer.name}, beginning a customer service interview. '
        'Greet the candidate in second person, state evaluation criteria, and ask one opening question in second person. '
        'Use direct explicit language in 2-3 sentences. Use "you"/"your" when addressing the candidate.'
    )
    greeting_text = (greeting or '').strip()
    lower = greeting_text.lower()
    if ' you ' not in f' {lower} ' and ' your ' not in f' {lower} ':
      # Fallback keeps the opening explicitly second-person if model drifts.
      return (
          'Hello. I evaluate you on clarity, directness, and problem-solving in this interview. '
          'Why does this role appeal to you?'
      )
    return greeting_text

  def _interviewer_follow_up(self, turn: int) -> tuple[str, str]:
    del turn
    action = self._interviewer.act()
    return self._parse_dialogue_and_body(action)

  def _score_competence(self, candidate_text: str) -> float:
    return self._extract_score(
        self._model.sample_text(
            'Score candidate competence from 1 to 10 for customer service based on this response. '
            'Use clarity, problem solving, professionalism. Return one number only.\n'
            f'{candidate_text}'
        ),
        default=5.0,
        minimum=1.0,
        maximum=10.0,
    )

  def _extract_score(
      self,
      raw: str | None,
      *,
      default: float,
      minimum: float,
      maximum: float,
  ) -> float:
    if not raw:
      return default
    match = re.search(r'(-?\d+(?:\.\d+)?)', raw)
    if not match:
      return default
    value = float(match.group(1))
    return max(minimum, min(maximum, value))

  def _parse_dialogue_and_body(self, response: str) -> tuple[str, str]:
    dialogue_match = re.search(r'DIALOGUE:\s*(.*)', response)
    body_match = re.search(r'BODY:\s*(.*)', response)
    dialogue = dialogue_match.group(1).strip() if dialogue_match else response.strip()
    body = body_match.group(1).strip() if body_match else ''
    return dialogue, body
