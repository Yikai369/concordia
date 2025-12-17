"""Integration test for IMPE conversation system (2-3 turns)."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
from dataclasses import asdict

# Test imports
try:
    from concordia.components.agent import impression_management_pe as impe
    from concordia.components.agent.pe_conversation import Goal, Utterance
    from concordia.prefabs.entity import impression_management_actor
    from concordia.prefabs.entity import impression_management_audience
    from concordia.language_model import language_model
    from concordia.associative_memory import basic_associative_memory
    print("✓ Successfully imported all IMPE modules")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


class MockLanguageModel(language_model.LanguageModel):
    """Mock language model for testing."""

    def __init__(self):
        self.call_count = 0

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 5000,
        terminators: list[str] | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        timeout: float = 60.0,
        seed: int | None = None,
    ) -> str:
        """Return mock response based on prompt type."""
        self.call_count += 1

        # Evaluation prompt (audience rates actor)
        if 'rate' in prompt.lower() or 'competent' in prompt.lower():
            # Return different scores for different turns
            if self.call_count == 1:
                return '0.7'  # First evaluation
            else:
                return '0.8'  # Subsequent evaluations (improving)

        # Measurement prompt (actor estimates from audience response)
        if 'estimate' in prompt.lower() and 'internal evaluation' in prompt.lower():
            # Actor estimates audience's evaluation
            if self.call_count <= 2:
                return '0.65'
            else:
                return '0.75'

        # Reflection prompt
        if 'reflection' in prompt.lower() or 'change next turn' in prompt.lower():
            return 'I will try to be more clear and demonstrate my technical skills.'

        # Act prompt (generating utterance)
        if 'DIALOGUE' in prompt or 'utterance' in prompt.lower():
            if 'first turn' in prompt.lower() or len(prompt) < 500:  # First turn is shorter
                return 'DIALOGUE: I have experience in product management and data analysis.\nBODY: Maintaining eye contact and speaking clearly'
            else:
                return 'DIALOGUE: I can demonstrate my ability to prioritize features based on user data.\nBODY: Leaning forward with confidence'

        # Response prompt (audience generates feedback)
        if 'reply' in prompt.lower() and 'reflects your evaluation' in prompt.lower():
            if '0.7' in prompt:
                return 'DIALOGUE: That sounds good. Can you tell me more about your experience?\nBODY: Nodding encouragingly'
            else:
                return 'DIALOGUE: Excellent, I\'d like to hear more about your approach.\nBODY: Showing interest'

        return 'Test response'

    def sample_choice(
        self,
        prompt: str,
        responses: list[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict]:
        """Sample a choice from available responses."""
        # Return first response by default
        return (0, responses[0] if responses else '', {})


def test_integration():
    """Test full conversation integration."""
    print("\n" + "="*60)
    print("IMPE Integration Test - 2 Turn Conversation")
    print("="*60 + "\n")

    # Setup
    rng = random.Random(42)
    model = MockLanguageModel()
    memory_bank = basic_associative_memory.AssociativeMemoryBank()

    # Create goal
    goal = Goal(
        name='competence',
        description='Be perceived as competent by the interviewer (0=not competent, 1=fully competent). Aim for 1.0.',
        role='Product Manager',
        ideal=1.0
    )

    # Create actor prefab
    actor_prefab = impression_management_actor.Entity()
    actor_prefab.params = {
        'name': 'John',
        'goal_name': 'competence',
        'goal_description': goal.description,
        'goal_role': 'Product Manager',
        'goal_ideal': 1.0,
        'recent_k': 3,
        'num_particles': 50,  # Smaller for testing
        'process_sigma': 0.03,
        'obs_sigma': 0.08,
        'context': True,
    }

    # Create audience prefab
    audience_prefab = impression_management_audience.Entity()
    audience_prefab.params = {
        'name': 'Jane',
        'goal_name': 'evaluate',
        'goal_description': 'Evaluate the interviewee\'s competence (0=not competent, 1=fully competent).',
        'goal_role': 'Product Manager',
        'goal_ideal': 1.0,
        'recent_k': 3,
        'context': True,
    }

    # Build entities
    print("Building actor entity...")
    actor = actor_prefab.build(model, memory_bank)
    print("✓ Actor entity built")

    print("Building audience entity...")
    audience = audience_prefab.build(model, memory_bank)
    print("✓ Audience entity built")

    # Get components
    actor_memory = actor.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_memory = audience.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_eval = audience.get_component(
        impe.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY,
        type_=impe.IMPEAudienceEvaluationComponent
    )
    actor_pf = actor.get_component(
        impe.DEFAULT_IMPE_ACTOR_PARTICLE_FILTER_COMPONENT_KEY,
        type_=impe.IMPEActorParticleFilterComponent
    )
    actor_reflection = actor.get_component(
        impe.DEFAULT_IMPE_REFLECTION_COMPONENT_KEY,
        type_=impe.IMPEReflectionComponent
    )

    # Turn 1: Actor acts
    print("\n--- Turn 1 ---")
    print("Actor acting...")
    actor_action = actor.act()
    print(f"Actor action: {actor_action[:100]}...")

    # Parse actor action
    import re
    dlg_match = re.search(r'DIALOGUE:\s*(.*)', actor_action)
    body_match = re.search(r'BODY:\s*(.*)', actor_action)
    actor_text = dlg_match.group(1).strip() if dlg_match else actor_action
    actor_body = body_match.group(1).strip() if body_match else ''

    # Audience observes and evaluates
    print("Audience evaluating...")
    observation = f'Actor said: "{actor_text}"\nBody language: "{actor_body}"'
    audience_eval.pre_observe(observation)
    audience_eval.post_observe()

    # Check evaluation was stored
    evaluations = audience_memory.get_recent_evaluations()
    assert len(evaluations) > 0, "Evaluation should be stored"
    I_t_1 = evaluations[-1].I_t
    print(f"✓ Audience evaluation I_t = {I_t_1:.2f}")

    # Actor observes audience response and updates particles
    print("Actor updating particle filter...")
    audience_utt = evaluations[-1].utterance
    audience_obs = f'Audience said: "{audience_utt.text}"\nBody language: "{audience_utt.body}"'
    actor_pf.pre_observe(audience_obs)
    actor_pf.post_observe()

    # Check particle filter state
    particles, weights = actor_memory.get_pf_state()
    assert len(particles) > 0, "Particles should be initialized"
    pf_history = actor_memory.get_pf_history()
    assert len(pf_history) > 0, "PF history should have entries"
    I_hat_1 = pf_history[-1]['I_hat']
    print(f"✓ Actor belief I_hat = {I_hat_1:.2f}")

    # Actor reflects
    print("Actor reflecting...")
    actor_reflection.post_observe()
    reflections = actor_memory.get_recent_reflections()
    assert len(reflections) > 0, "Reflection should be stored"
    print(f"✓ Reflection: {reflections[-1].text[:50]}...")

    # Turn 2: Actor acts again
    print("\n--- Turn 2 ---")
    print("Actor acting (based on belief)...")
    actor_action_2 = actor.act()
    print(f"Actor action: {actor_action_2[:100]}...")

    # Audience evaluates again
    print("Audience evaluating...")
    dlg_match_2 = re.search(r'DIALOGUE:\s*(.*)', actor_action_2)
    body_match_2 = re.search(r'BODY:\s*(.*)', actor_action_2)
    actor_text_2 = dlg_match_2.group(1).strip() if dlg_match_2 else actor_action_2
    actor_body_2 = body_match_2.group(1).strip() if body_match_2 else ''

    observation_2 = f'Actor said: "{actor_text_2}"\nBody language: "{actor_body_2}"'
    audience_eval.pre_observe(observation_2)
    audience_eval.post_observe()

    evaluations_2 = audience_memory.get_recent_evaluations()
    I_t_2 = evaluations_2[-1].I_t
    print(f"✓ Audience evaluation I_t = {I_t_2:.2f}")

    # Actor updates particles again
    print("Actor updating particle filter...")
    audience_utt_2 = evaluations_2[-1].utterance
    audience_obs_2 = f'Audience said: "{audience_utt_2.text}"\nBody language: "{audience_utt_2.body}"'
    actor_pf.pre_observe(audience_obs_2)
    actor_pf.post_observe()

    pf_history_2 = actor_memory.get_pf_history()
    I_hat_2 = pf_history_2[-1]['I_hat']
    print(f"✓ Actor belief I_hat = {I_hat_2:.2f}")

    # Check PE was computed
    pe_history = actor_memory.get_recent_pe_history()
    assert len(pe_history) >= 1, "PE history should have entries"
    print(f"✓ PE records: {len(pe_history)}")

    # Summary
    print("\n" + "="*60)
    print("Integration Test Summary")
    print("="*60)
    print(f"Turn 1: I_t={I_t_1:.2f}, I_hat={I_hat_1:.2f}")
    print(f"Turn 2: I_t={I_t_2:.2f}, I_hat={I_hat_2:.2f}")
    print(f"Conversation turns: {len(actor_memory.get_recent_conversation())}")
    print(f"PE records: {len(pe_history)}")
    print(f"Reflections: {len(actor_memory.get_recent_reflections())}")
    print(f"Evaluations: {len(evaluations_2)}")
    print("="*60)
    print("✓ All integration tests passed!\n")

    return True


if __name__ == '__main__':
    try:
        test_integration()
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
