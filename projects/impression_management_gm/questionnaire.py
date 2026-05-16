"""Questionnaire utilities for setup-only validation of IMPE agents."""

from __future__ import annotations

import json
import re
from typing import Any

from concordia.language_model import language_model

from projects.impression_management_gm import constants

PERTH_ALEXITHYMIA_QUESTIONS = [
    'When I’m feeling bad (feeling an unpleasant emotion), I can’t find the right words to describe those feelings.',
    'When I’m feeling bad, I can’t tell whether I’m sad, angry, or scared.',
    'I tend to ignore how I feel.',
    'When I’m feeling good (feeling a pleasant emotion), I can’t find the right words to describe those feelings.',
    'When I’m feeling good, I can’t tell whether I’m happy, excited, or amused.',
    'I prefer to just let my feelings happen in the background, rather than focus on them.',
    'When I’m feeling bad, I can’t talk about those feelings in much depth or detail.',
    'When I’m feeling bad, I can’t make sense of those feelings.',
    'I don’t pay attention to my emotions.',
    'When I’m feeling good, I can’t talk about those feelings in much depth or detail.',
    'When I’m feeling good, I can’t make sense of those feelings.',
    'Usually, I try to avoid thinking about what I’m feeling.',
    'When something bad happens, it’s hard for me to put into words how I’m feeling.',
    'When I’m feeling bad, I get confused about what emotion it is.',
    'When I’m feeling bad, I get confused about what emotion it is.',
    'When something good happens, it’s hard for me to put into words how I’m feeling.',
    'When I’m feeling good, I get confused about what emotion it is.',
    'I don’t try to be "in touch" with my emotions.',
    'When I’m feeling bad, if I try to describe how I’m feeling I don’t know what to say.',
    'When I’m feeling bad, I’m puzzled by those feelings.',
    'It’s not important for me to know what I’m feeling.',
    'When I’m feeling good, if I try to describe how I’m feeling I don’t know what to say.',
    'When I’m feeling good, I’m puzzled by those feelings.',
    'It’s strange for me to think about my emotions.',
]


PERTH_ALEXITHYMIA_SCALE = [
    ('A', 'Strongly agree'),
    ('B', 'Moderately agree'),
    ('C', 'Slightly agree'),
    ('D', 'Neutral'),
    ('E', 'Slightly disagree'),
    ('F', 'Moderately disagree'),
    ('G', 'Strongly disagree'),
]

EMPATHY_QUESTIONS = [
    'I daydream and fantasize, with some regularity, about things that might happen to me.',
    'I often have tender, concerned feelings for people less fortunate than me.',
    'I sometimes find it difficult to see things from the "other guy\'s" point of view.',
    'Sometimes I don\'t feel very sorry for other people when they are having problems.',
    'I really get involved with the feelings of the characters in a novel.',
    'In emergency situations, I feel apprehensive and ill-at-ease.',
    'I am usually objective when I watch a movie or play, and I don\'t often get completely caught up in it.',
    'I try to look at everybody\'s side of a disagreement before I make a decision.',
    'When I see someone being taken advantage of, I feel kind of protective towards them.',
    'I sometimes feel helpless when I am in the middle of a very emotional situation.',
    'I sometimes try to understand my friends better by imagining how things look from their perspective.',
    'Becoming extremely involved in a good book or movie is somewhat rare for me.',
    'When I see someone get hurt, I tend to remain calm.',
    'Other people\'s misfortunes do not usually disturb me a great deal.',
    'If I\'m sure I\'m right about something, I don\'t waste much time listening to other people\'s arguments.',
    'After seeing a play or movie, I have felt as though I were one of the characters.',
    'Being in a tense emotional situation scares me.',
    'When I see someone being treated unfairly, I sometimes don\'t feel very much pity for them.',
    'I am usually pretty effective in dealing with emergencies.',
    'I am often quite touched by things that I see happen.',
    'I believe that there are two sides to every question and try to look at them both.',
    'I would describe myself as a pretty soft-hearted person.',
    'When I watch a good movie, I can very easily put myself in the place of a leading character.',
    'I tend to lose control during emergencies.',
    'When I\'m upset at someone, I usually try to "put myself in his shoes" for a while.',
    'When I am reading an interesting story or novel, I imagine how I would feel if the events in the story were happening to me.',
    'When I see someone who badly needs help in an emergency, I go to pieces.',
    'Before criticizing somebody, I try to imagine how I would feel if I were in their place.',
]

EMPATHY_SCALE = [
    ('1', 'Does not describe me well'),
    ('2', 'Describes me slightly'),
    ('3', 'Describes me moderately'),
    ('4', 'Describes me well'),
    ('5', 'Describes me very well'),
]

CV1_CHUNKS: list[list[tuple[int, str]]] = [
  [
    (2, 'When I’m feeling bad, I can’t tell whether I’m sad, angry, or scared.'),
    (8, 'When I’m feeling bad, I can’t make sense of those feelings.'),
    (14, 'When I’m feeling bad, I get confused about what emotion it is.'),
    (20, 'When I’m feeling bad, I’m puzzled by those feelings.'),
  ],
  [
    (5, 'When I’m feeling good, I can’t tell whether I’m happy, excited, or amused.'),
    (11, 'When I’m feeling good, I can’t make sense of those feelings.'),
    (17, 'When I’m feeling good, I get confused about what emotion it is.'),
    (23, 'When I’m feeling good, I’m puzzled by those feelings.'),
  ],
  [
    (1, 'When I’m feeling bad (feeling an unpleasant emotion), I can’t find the right words to describe those feelings.'),
    (7, 'When I’m feeling bad, I can’t talk about those feelings in much depth or detail.'),
    (13, 'When something bad happens, it’s hard for me to put into words how I’m feeling.'),
    (19, 'When I’m feeling bad, if I try to describe how I’m feeling I don’t know what to say.'),
  ],
  [
    (4, 'When I’m feeling good (feeling a pleasant emotion), I can’t find the right words to describe those feelings.'),
    (10, 'When I’m feeling good, I can’t talk about those feelings in much depth or detail.'),
    (16, 'When something good happens, it’s hard for me to put into words how I’m feeling.'),
    (22, 'When I’m feeling good, if I try to describe how I’m feeling I don’t know what to say.'),
  ],
  [
    (3, 'I tend to ignore how I feel.'),
    (6, 'I prefer to just let my feelings happen in the background, rather than focus on them.'),
    (9, 'I don’t pay attention to my emotions.'),
    (12, 'Usually, I try to avoid thinking about what I’m feeling.'),
    (15, 'When I’m feeling bad, I get confused about what emotion it is.'),
    (18, 'I don’t try to be "in touch" with my emotions.'),
    (21, 'It’s not important for me to know what I’m feeling.'),
    (24, 'It’s strange for me to think about my emotions.'),
  ],
]


def run_convergent_validity_questionnaire(
    *,
    model: language_model.LanguageModel,
    agent_name: str,
    role_label: str,
    neurotype: str,
    role_context: str,
    memories: list[str],
) -> dict[str, Any]:
  chunk_results: list[dict[str, Any]] = []
  merged_answers: dict[int, str] = {}

  for chunk in CV1_CHUNKS:
    question_ids = [item for item, _ in chunk]
    questions = [text for _, text in chunk]
    chunk_result = _run_questionnaire(
        model=model,
        agent_name=agent_name,
        role_label=role_label,
        neurotype=neurotype,
        role_context=role_context,
        memories=memories,
        questions=questions,
        question_ids=question_ids,
        scale=PERTH_ALEXITHYMIA_SCALE,
        questionnaire_name='CV1',
        answer_format='A-G',
    )
    chunk_results.append(chunk_result)
    for answer in chunk_result.get('parsed_response', {}).get('answers', []):
      item = answer.get('item')
      value = answer.get('answer')
      if isinstance(item, int) and isinstance(value, str):
        merged_answers[item] = value

  ordered_answers = [
      {'item': item, 'answer': merged_answers[item]}
      for item in sorted(merged_answers)
  ]
  raw_chunks = [
      {
          'chunk': index,
          'raw_response': result.get('raw_response', ''),
      }
      for index, result in enumerate(chunk_results, start=1)
  ]

  return {
      'agent_name': agent_name,
      'role_label': role_label,
      'neurotype': neurotype,
      'questionnaire_name': 'CV1',
      'raw_response': json.dumps(raw_chunks, ensure_ascii=False),
      'parsed_response': {
        'raw': json.dumps(raw_chunks, ensure_ascii=False),
        'answers': ordered_answers,
      },
      'question_count': len(ordered_answers),
    }


def run_empathy_questionnaire(
    *,
    model: language_model.LanguageModel,
    agent_name: str,
    role_label: str,
    neurotype: str,
    role_context: str,
    memories: list[str],
) -> dict[str, Any]:
  return _run_questionnaire(
      model=model,
      agent_name=agent_name,
      role_label=role_label,
      neurotype=neurotype,
      role_context=role_context,
      memories=memories,
      questions=EMPATHY_QUESTIONS,
      question_ids=None,
      scale=EMPATHY_SCALE,
      questionnaire_name='CV2',
      answer_format='1-5',
  )


def _run_questionnaire(
    *,
    model: language_model.LanguageModel,
    agent_name: str,
    role_label: str,
    neurotype: str,
    role_context: str,
    memories: list[str],
    questions: list[str],
    question_ids: list[int] | None,
    scale: list[tuple[str, str]],
    questionnaire_name: str,
    answer_format: str,
) -> dict[str, Any]:
  prompt = _build_questionnaire_prompt(
      agent_name=agent_name,
      role_label=role_label,
      neurotype=neurotype,
      role_context=role_context,
      memories=memories,
      questions=questions,
      question_ids=question_ids,
      scale=scale,
        answer_format=answer_format,
  )
  raw = (model.sample_text(prompt) or '').strip()
  parsed = _parse_questionnaire_response(raw)
  return {
      'agent_name': agent_name,
      'role_label': role_label,
      'neurotype': neurotype,
      'questionnaire_name': questionnaire_name,
      'raw_response': raw,
      'parsed_response': parsed,
      'question_count': len(questions),
  }


def _build_questionnaire_prompt(
    *,
    agent_name: str,
    role_label: str,
    neurotype: str,
    role_context: str,
    memories: list[str],
    questions: list[str],
    question_ids: list[int] | None,
    scale: list[tuple[str, str]],
    answer_format: str,
) -> str:
  norms_text = _format_norms(constants.ALL_CULTURAL_NORMS)
  trait_paragraph = constants.NEUROTYPE_TRAIT_PARAGRAPHS.get(
      neurotype,
      constants.NEUROTYPE_TRAIT_PARAGRAPHS[constants.NEUROTYPE_CADEN],
  ).strip()
  memory_text = '\n'.join(f'- {memory}' for memory in memories) or '- (none)'
  if question_ids is None:
    question_ids = list(range(1, len(questions) + 1))
  questions_text = '\n'.join(
      f'{item}. {question}'
      for item, question in zip(question_ids, questions)
  )
  scale_text = '\n'.join(f'{symbol}) {label}' for symbol, label in scale)
  return f"""You are answering as {agent_name}.

Use only the information below:

- Shared communication norms:
{norms_text}

- Neurotype context:
{trait_paragraph}

- Formative memories:
{memory_text}

Answer the questionnaire as this person would answer it in a standard interaction setting.
Do not explain your choices.
Answer each item independently. Do not choose one global response and reuse it for every item.
Base each answer on the specific statement being asked, even if several items seem similar.

Answer format: {answer_format}

For each item, choose exactly one response from this scale:
{scale_text}

Questionnaire items:
{questions_text}

Return JSON only in this exact shape:
{{"answers":[{{"item":{question_ids[0]},"answer":"{scale[0][0]}"}}]}}

Include one entry per item, in order.
Use only the answer symbol.
Do not wrap the JSON in markdown fences.
"""


def _format_norms(norms: list[Any]) -> str:
  return '\n'.join(
      f'- {norm.name}: {norm.description}'
      for norm in norms
  )


def _parse_questionnaire_response(raw: str) -> dict[str, Any]:
  if not raw:
    return {'raw': '', 'answers': []}

  json_text = raw
  block_match = re.search(r'\{.*\}', raw, flags=re.DOTALL)
  if block_match:
    json_text = block_match.group(0)

  try:
    data = json.loads(json_text)
  except json.JSONDecodeError:
    return {'raw': raw, 'answers': []}

  answers = data.get('answers')
  if not isinstance(answers, list):
    return {'raw': raw, 'answers': []}

  normalized_answers: list[dict[str, Any]] = []
  for entry in answers:
    if not isinstance(entry, dict):
      continue
    item = entry.get('item')
    answer = entry.get('answer')
    if not isinstance(item, int):
      continue
    if not isinstance(answer, str):
      continue
    # Keep answer as-is (could be letter or number)
    normalized_answers.append({'item': item, 'answer': answer.strip()})

  return {
      'raw': raw,
      'answers': normalized_answers,
  }
