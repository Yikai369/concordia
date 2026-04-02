"""Entity builders for the IM game-master experiment.

Refactors and reuses Concordia IMPE actor and audience prefabs while adapting
them to candidate/interviewer roles in the im_gm specification.
"""

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity import impression_management_actor
from concordia.prefabs.entity import impression_management_audience

from projects.impression_management_gm import constants


def _traits_for_neurotype(neurotype: str) -> str:
  trait_paragraph = constants.NEUROTYPE_TRAIT_PARAGRAPHS.get(
      neurotype,
      constants.NEUROTYPE_TRAIT_PARAGRAPHS[constants.NEUROTYPE_CADEN],
  )
  group_context = constants.GROUP_BEHAVIOR_CONTEXT.get(
      neurotype,
      constants.GROUP_BEHAVIOR_CONTEXT[constants.NEUROTYPE_CADEN],
  )
  return f'{group_context}\n\n{trait_paragraph}'


def build_candidate_and_interviewer(
    model: language_model.LanguageModel,
    memory_bank: basic_associative_memory.AssociativeMemoryBank,
    candidate_name: str,
    interviewer_name: str,
    candidate_neurotype: str,
    interviewer_neurotype: str,
):
  """Build IMPE candidate (actor prefab) and interviewer (audience prefab)."""

  candidate_prefab = impression_management_actor.Entity()
  candidate_prefab.params = {
      'name': candidate_name,
      'goal_name': 'competence',
      'goal_description': (
          'Be perceived as competent for the customer service role.'
      ),
      'goal_role': 'Customer Service Representative',
      'recent_k': constants.DEFAULT_RECENT_K,
      'num_particles': constants.DEFAULT_NUM_PARTICLES,
      'process_sigma': constants.DEFAULT_PROCESS_SIGMA,
      'obs_sigma': constants.DEFAULT_OBS_SIGMA,
      'context': True,
      'cultural_norms': constants.ALL_CULTURAL_NORMS,
      'traits_paragraph': _traits_for_neurotype(candidate_neurotype),
      'enable_world_building': True,
      'enable_interview_context': True,
      'use_full_2a25_world': True,
      'enable_instructions': True,
      'enable_self_perception': True,
      'enable_situation_perception': False,
      'enable_person_by_situation': False,
      'use_option_space': True,
      'use_memory_check': True,
      'enable_self_assessment': True,
      'consistency_threshold': 0.68,
      'disable_revision': False,
  }

  interviewer_prefab = impression_management_audience.Entity()
  interviewer_prefab.params = {
      'name': interviewer_name,
      'goal_name': 'evaluate',
      'goal_description': (
          'Evaluate candidate competence and role fit.'
      ),
      'goal_role': 'Customer Service Representative',
      'recent_k': constants.DEFAULT_RECENT_K,
      'context': True,
      'cultural_norms': constants.ALL_CULTURAL_NORMS,
      'traits_paragraph': _traits_for_neurotype(interviewer_neurotype),
      'enable_world_building': True,
      'enable_interview_context': True,
      'use_full_2a25_world': True,
      'use_memory_check': True,
  }

  candidate = candidate_prefab.build(model, memory_bank)
  interviewer = interviewer_prefab.build(model, memory_bank)
  return candidate, interviewer
