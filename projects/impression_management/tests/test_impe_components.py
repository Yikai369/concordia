"""Simple standalone test for IMPE components."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import unittest

# Test imports
try:
    from concordia.components.agent import impression_management_pe as impe
    from concordia.components.agent.pe_conversation import Goal, Utterance
    print("✓ Successfully imported IMPE components")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class ParticleFilterTest(unittest.TestCase):
    """Test ParticleFilter class."""

    def test_initialize(self):
        """Test particle filter initialization."""
        pf = impe.ParticleFilter(num_particles=10, rng=random.Random(42))
        particles, weights = pf.initialize()
        self.assertEqual(len(particles), 10)
        self.assertEqual(len(weights), 10)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        # All particles should be in [0, 1]
        for p in particles:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        print("✓ ParticleFilter.initialize() test passed")

    def test_predict(self):
        """Test prediction step."""
        pf = impe.ParticleFilter(num_particles=10, rng=random.Random(42))
        particles, _ = pf.initialize([0.5] * 10)
        predicted = pf.predict(particles)
        self.assertEqual(len(predicted), 10)
        # Particles should still be in [0, 1]
        for p in predicted:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        print("✓ ParticleFilter.predict() test passed")

    def test_update(self):
        """Test update step."""
        pf = impe.ParticleFilter(num_particles=10, rng=random.Random(42))
        particles, _ = pf.initialize([0.5] * 10)
        updated_particles, updated_weights, ess, resampled = pf.update(particles, 0.7)
        self.assertEqual(len(updated_particles), 10)
        self.assertEqual(len(updated_weights), 10)
        self.assertIsInstance(ess, float)
        self.assertIsInstance(resampled, bool)
        print("✓ ParticleFilter.update() test passed")


class IMPEMemoryComponentTest(unittest.TestCase):
    """Test IMPEMemoryComponent."""

    def test_add_evaluation_record(self):
        """Test adding evaluation records."""
        goal = Goal(name='test', description='test goal')
        memory = impe.IMPEMemoryComponent(goal=goal)
        utterance = Utterance(turn=1, speaker='Jane', text='Hello', body='smiling')
        memory.add_evaluation_record(1, 0.75, utterance)
        evaluations = memory.get_recent_evaluations()
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0].I_t, 0.75)
        self.assertEqual(evaluations[0].turn, 1)
        print("✓ IMPEMemoryComponent.add_evaluation_record() test passed")

    def test_particle_filter_state(self):
        """Test particle filter state management."""
        goal = Goal(name='test', description='test goal')
        memory = impe.IMPEMemoryComponent(goal=goal)
        particles = [0.5] * 10
        weights = [0.1] * 10
        history_entry = {'turn': 1, 'I_hat': 0.5, 'ess': 10.0}
        memory.update_particle_filter_state(particles, weights, history_entry)
        stored_particles, stored_weights = memory.get_pf_state()
        self.assertEqual(len(stored_particles), 10)
        self.assertEqual(len(stored_weights), 10)
        pf_history = memory.get_pf_history()
        self.assertEqual(len(pf_history), 1)
        self.assertEqual(pf_history[0]['I_hat'], 0.5)
        print("✓ IMPEMemoryComponent particle filter state test passed")


class CulturalNormsComponentTest(unittest.TestCase):
    """Test CulturalNormsComponent."""

    def test_get_norms_text(self):
        """Test formatting norms text."""
        norms = [
            impe.CulturalNorm('Norm1', 'Description1'),
            impe.CulturalNorm('Norm2', 'Description2'),
        ]
        comp = impe.CulturalNormsComponent(norms=norms)
        text = comp.get_norms_text()
        self.assertIn('CULTURAL NORMS YOU FOLLOW', text)
        self.assertIn('Norm1', text)
        self.assertIn('Description1', text)
        print("✓ CulturalNormsComponent.get_norms_text() test passed")

    def test_empty_norms(self):
        """Test with no norms."""
        comp = impe.CulturalNormsComponent(norms=[])
        text = comp.get_norms_text()
        self.assertEqual(text, '')
        print("✓ CulturalNormsComponent empty norms test passed")


class PersonalityTraitsComponentTest(unittest.TestCase):
    """Test PersonalityTraitsComponent."""

    def test_get_traits_text(self):
        """Test formatting traits text."""
        traits = [
            impe.PersonalityTrait('Trait1', 'Assertion1'),
        ]
        trait_scores = {'Trait1': 2}
        comp = impe.PersonalityTraitsComponent(traits=traits, trait_scores=trait_scores)
        text = comp.get_traits_text()
        self.assertIn('PERSONALITY TRAITS', text)
        self.assertIn('Trait1', text)
        self.assertIn('(2/3)', text)
        print("✓ PersonalityTraitsComponent.get_traits_text() test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Running IMPE Component Tests")
    print("="*60 + "\n")
    unittest.main(verbosity=2)
