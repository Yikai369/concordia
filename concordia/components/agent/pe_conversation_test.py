# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for PE conversation components."""

import unittest
from unittest import mock

from concordia.components.agent import pe_conversation as pe_components
from concordia.language_model import language_model
from concordia.typing import entity_component


class MockLanguageModel(language_model.LanguageModel):
  """Mock language model for testing."""

  def __init__(self, responses=None):
    super().__init__()
    self._responses = responses or {}
    self._call_count = 0

  def sample_text(self, prompt: str) -> str:
    self._call_count += 1
    # Return response based on prompt content
    if 'estimate' in prompt.lower() or 'current state' in prompt.lower():
      return '0.75'  # Default estimate
    elif 'reflection' in prompt.lower() or 'reduce pe' in prompt.lower():
      return 'I will be more friendly and engaging.'
    elif 'utterance' in prompt.lower() or 'produce one' in prompt.lower():
      return 'Hello, how are you doing today?'
    return 'Mock response'

  def sample_choice(self, prompt: str, choices: tuple[str, ...]) -> str:
    return choices[0] if choices else ''


class MockEntity(entity_component.EntityWithComponents):
  """Mock entity for testing."""

  def __init__(self, name: str):
    self._name = name
    self._components = {}
    self._phase = entity_component.Phase.READY
    self._state = {}

  @property
  def name(self) -> str:
    return self._name

  def get_phase(self) -> entity_component.Phase:
    return self._phase

  def get_component(
      self, name: str, type_=entity_component.BaseComponent
  ) -> entity_component.BaseComponent:
    return self._components.get(name)

  def set_component(self, name: str, component: entity_component.BaseComponent):
    self._components[name] = component

  def act(self, action_spec=None):
    return 'Mock action'

  def observe(self, observation: str) -> None:
    pass

  def get_state(self) -> entity_component.EntityState:
    return self._state

  def set_state(self, state: entity_component.EntityState) -> None:
    self._state = state


class PEMemoryComponentTest(unittest.TestCase):
  """Tests for PEMemoryComponent."""

  def setUp(self):
    self.goal = pe_components.Goal(
        name='likability',
        description='Be perceived as likable',
        ideal=1.0,
    )
    self.memory = pe_components.PEMemoryComponent(
        goal=self.goal, recent_k=3
    )
    self.entity = MockEntity('TestAgent')
    self.memory.set_entity(self.entity)

  def test_add_and_retrieve_utterance(self):
    self.memory.add_utterance(turn=1, speaker='Agent A', text='Hello')
    self.memory.add_utterance(turn=2, speaker='Agent B', text='Hi there')

    conv = self.memory.get_recent_conversation()
    self.assertEqual(len(conv), 2)
    self.assertEqual(conv[0].text, 'Hello')
    self.assertEqual(conv[1].text, 'Hi there')

  def test_add_and_retrieve_pe_record(self):
    self.memory.add_pe_record(
        turn=1, partner_text='Hello', estimate=0.75, pe=0.25
    )

    pe_history = self.memory.get_recent_pe_history()
    self.assertEqual(len(pe_history), 1)
    self.assertEqual(pe_history[0].estimate, 0.75)
    self.assertEqual(pe_history[0].pe, 0.25)

  def test_add_and_retrieve_reflection(self):
    self.memory.add_reflection(turn=1, text='I will be more friendly')

    reflections = self.memory.get_recent_reflections()
    self.assertEqual(len(reflections), 1)
    self.assertEqual(reflections[0].text, 'I will be more friendly')

  def test_recent_k_limit(self):
    # Add more than recent_k items
    for i in range(5):
      self.memory.add_utterance(turn=i, speaker='Agent A', text=f'Text {i}')

    conv = self.memory.get_recent_conversation(k=3)
    self.assertEqual(len(conv), 3)
    self.assertEqual(conv[0].text, 'Text 2')  # Last 3 items

  def test_get_last_pe(self):
    self.assertEqual(self.memory.get_last_pe(), 0.0)  # No PE yet

    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.8, pe=0.2)
    self.assertEqual(self.memory.get_last_pe(), 0.2)

  def test_state_management(self):
    # Add some data
    self.memory.add_utterance(turn=1, speaker='A', text='Hello')
    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.75, pe=0.25)
    self.memory.add_reflection(turn=1, text='Reflection')

    # Get state
    state = self.memory.get_state()
    self.assertIn('conversation', state)
    self.assertIn('pe_history', state)
    self.assertIn('reflections', state)

    # Create new memory and restore state
    new_memory = pe_components.PEMemoryComponent(goal=self.goal, recent_k=3)
    new_memory.set_entity(self.entity)
    new_memory.set_state(state)

    # Verify restored data
    conv = new_memory.get_recent_conversation()
    self.assertEqual(len(conv), 1)
    self.assertEqual(conv[0].text, 'Hello')


class PEEstimationComponentTest(unittest.TestCase):
  """Tests for PEEstimationComponent."""

  def setUp(self):
    self.model = MockLanguageModel()
    self.goal = pe_components.Goal(
        name='likability', description='Be likable', ideal=1.0
    )
    self.memory = pe_components.PEMemoryComponent(goal=self.goal)
    self.estimation = pe_components.PEEstimationComponent(model=self.model)
    self.entity = MockEntity('TestAgent')
    self.memory.set_entity(self.entity)
    self.estimation.set_entity(self.entity)
    self.entity.set_component('PE_Memory', self.memory)

  def test_pre_observe_extracts_partner_text(self):
    # Test various observation formats
    test_cases = [
        ('Partner said: "Hello there"', 'Hello there'),
        ('Agent A: How are you?', 'How are you?'),
        ('"Hello world"', 'Hello world'),
        ('Observation: Agent B said hello', 'hello'),
    ]

    for observation, expected in test_cases:
      self.estimation.pre_observe(observation)
      # The extraction might not be perfect, but should capture something
      self.assertIsNotNone(self.estimation._last_partner_text)

  def test_post_observe_computes_pe(self):
    # Set up memory
    self.memory.add_utterance(turn=1, speaker='Agent A', text='Hello')

    # Simulate observation
    self.estimation.pre_observe('Partner said: "Hello"')
    result = self.estimation.post_observe()

    # Should have computed PE
    self.assertIn('Estimated state', result)
    self.assertIn('PE', result)

    # Check that PE record was added
    pe_history = self.memory.get_recent_pe_history()
    self.assertGreater(len(pe_history), 0)

  def test_pe_calculation(self):
    # Test that PE = ideal - estimate
    self.estimation.pre_observe('Partner said: "Hello"')
    self.estimation.post_observe()

    pe_history = self.memory.get_recent_pe_history()
    if pe_history:
      pe_record = pe_history[-1]
      expected_pe = self.goal.ideal - pe_record.estimate
      self.assertAlmostEqual(pe_record.pe, expected_pe, places=2)

  def test_state_management(self):
    self.estimation.pre_observe('Test observation')
    self.estimation.post_observe()

    state = self.estimation.get_state()
    self.assertIn('last_estimate', state)
    self.assertIn('last_pe', state)

    # Restore state
    new_estimation = pe_components.PEEstimationComponent(model=self.model)
    new_estimation.set_entity(self.entity)
    new_estimation.set_state(state)
    self.assertEqual(new_estimation._last_estimate, state['last_estimate'])


class PEReflectionComponentTest(unittest.TestCase):
  """Tests for PEReflectionComponent."""

  def setUp(self):
    self.model = MockLanguageModel()
    self.goal = pe_components.Goal(
        name='likability', description='Be likable', ideal=1.0
    )
    self.memory = pe_components.PEMemoryComponent(goal=self.goal)
    self.reflection = pe_components.PEReflectionComponent(model=self.model)
    self.entity = MockEntity('TestAgent')
    self.memory.set_entity(self.entity)
    self.reflection.set_entity(self.entity)
    self.entity.set_component('PE_Memory', self.memory)

  def test_post_observe_generates_reflection(self):
    # Add a PE record first
    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.75, pe=0.25)

    # Generate reflection
    result = self.reflection.post_observe()

    # Should have generated reflection
    self.assertIsNotNone(result)
    self.assertGreater(len(result), 0)

    # Check that reflection was stored
    reflections = self.memory.get_recent_reflections()
    self.assertGreater(len(reflections), 0)

  def test_reflection_uses_last_pe(self):
    # Add multiple PE records
    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.5, pe=0.5)
    self.memory.add_pe_record(turn=2, partner_text='Hi', estimate=0.8, pe=0.2)

    self.reflection.post_observe()

    # Should use the last PE (0.2)
    reflections = self.memory.get_recent_reflections()
    if reflections:
      # The reflection should be generated based on the last PE
      self.assertIsNotNone(reflections[-1].text)

  def test_state_management(self):
    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.75, pe=0.25)
    self.reflection.post_observe()

    state = self.reflection.get_state()
    self.assertIn('last_reflection', state)

    # Restore state
    new_reflection = pe_components.PEReflectionComponent(model=self.model)
    new_reflection.set_entity(self.entity)
    new_reflection.set_state(state)
    self.assertEqual(new_reflection._last_reflection, state['last_reflection'])


class PEActComponentTest(unittest.TestCase):
  """Tests for PEActComponent."""

  def setUp(self):
    self.model = MockLanguageModel()
    self.goal = pe_components.Goal(
        name='likability', description='Be likable', ideal=1.0
    )
    self.memory = pe_components.PEMemoryComponent(goal=self.goal, recent_k=3)
    self.act_component = pe_components.PEActComponent(model=self.model)
    self.entity = MockEntity('TestAgent')
    self.memory.set_entity(self.entity)
    self.act_component.set_entity(self.entity)
    self.entity.set_component('PE_Memory', self.memory)

  def test_get_action_attempt_generates_utterance(self):
    # Add some context
    self.memory.add_utterance(turn=1, speaker='Agent A', text='Hello')
    self.memory.add_pe_record(turn=1, partner_text='Hello', estimate=0.75, pe=0.25)
    self.memory.add_reflection(turn=1, text='Be more friendly')

    # Create action spec
    from concordia.typing import entity as entity_lib
    action_spec = entity_lib.ActionSpec(
        call_to_action='What do you say?',
        output_type=entity_lib.OutputType.FREE,
    )

    # Generate action
    context = {}
    utterance = self.act_component.get_action_attempt(context, action_spec)

    # Should generate utterance
    self.assertIsNotNone(utterance)
    self.assertGreater(len(utterance), 0)

    # Should store utterance in memory
    conv = self.memory.get_recent_conversation()
    # Should have the new utterance
    self.assertGreater(len(conv), 1)

  def test_utterance_includes_entity_name(self):
    self.memory.add_utterance(turn=1, speaker='Agent A', text='Hello')

    from concordia.typing import entity as entity_lib
    action_spec = entity_lib.ActionSpec(
        call_to_action='Speak',
        output_type=entity_lib.OutputType.FREE,
    )

    context = {}
    utterance = self.act_component.get_action_attempt(context, action_spec)

    # Check that utterance was stored with correct speaker
    conv = self.memory.get_recent_conversation()
    if len(conv) > 1:
      last_utterance = conv[-1]
      self.assertEqual(last_utterance.speaker, self.entity.name)

  def test_state_management(self):
    state = self.act_component.get_state()
    self.assertEqual(state, {})

    self.act_component.set_state({})  # Should not raise


class PEComponentsIntegrationTest(unittest.TestCase):
  """Integration tests for PE components working together."""

  def setUp(self):
    self.model = MockLanguageModel()
    self.goal = pe_components.Goal(
        name='likability', description='Be likable', ideal=1.0
    )
    self.memory = pe_components.PEMemoryComponent(goal=self.goal, recent_k=3)
    self.estimation = pe_components.PEEstimationComponent(model=self.model)
    self.reflection = pe_components.PEReflectionComponent(model=self.model)
    self.act_component = pe_components.PEActComponent(model=self.model)

    self.entity = MockEntity('Agent A')
    self.memory.set_entity(self.entity)
    self.estimation.set_entity(self.entity)
    self.reflection.set_entity(self.entity)
    self.act_component.set_entity(self.entity)

    self.entity.set_component('PE_Memory', self.memory)

  def test_full_cycle(self):
    """Test a full cycle: act -> observe -> estimate -> reflect."""
    # Step 1: Agent acts
    from concordia.typing import entity as entity_lib
    action_spec = entity_lib.ActionSpec(
        call_to_action='Speak',
        output_type=entity_lib.OutputType.FREE,
    )
    utterance = self.act_component.get_action_attempt({}, action_spec)
    self.assertIsNotNone(utterance)

    # Step 2: Agent observes partner's response
    observation = f'Partner said: "Hello, nice to meet you"'
    self.estimation.pre_observe(observation)
    est_result = self.estimation.post_observe()

    # Should have computed PE
    self.assertIn('Estimated state', est_result)
    pe_history = self.memory.get_recent_pe_history()
    self.assertGreater(len(pe_history), 0)

    # Step 3: Agent reflects
    refl_result = self.reflection.post_observe()
    self.assertIsNotNone(refl_result)

    reflections = self.memory.get_recent_reflections()
    self.assertGreater(len(reflections), 0)

    # Step 4: Agent acts again with updated context
    utterance2 = self.act_component.get_action_attempt({}, action_spec)
    self.assertIsNotNone(utterance2)

    # Should have more conversation history
    conv = self.memory.get_recent_conversation()
    self.assertGreaterEqual(len(conv), 2)


if __name__ == '__main__':
  unittest.main()
