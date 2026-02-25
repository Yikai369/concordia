"""Configuration parsing and validation."""

import argparse
import os
import sys

from projects.impression_management_standard import constants
from projects.impression_management_standard.models import ConversationConfig
from projects.impression_management_standard import utils


def parse_arguments() -> ConversationConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Impression Management PE conversation (Standard Simulation Loop).'
    )
    parser.add_argument(
        '--turns',
        type=int,
        default=2,
        help='Total turns in dialogue.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini',
        help='OpenAI model name.',
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
        help='JSON output filename.',
    )
    parser.add_argument(
        '--no_audience_norms',
        action='store_true',
        help='Disable cultural norms for audience.',
    )
    parser.add_argument(
        '--no_traits',
        action='store_true',
        help='Disable personality traits.',
    )
    parser.add_argument(
        '--traits_file',
        type=str,
        default=None,
        help='Load personality traits from Excel (.xlsx) or CSV file with columns "name" and "assertion". Ignored if --no_traits.',
    )
    parser.add_argument(
        '--no_context',
        action='store_true',
        help='Disable interview context.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=7,
        help='Random seed for reproducibility.',
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Output directory (creates timestamped if None).',
    )
    parser.add_argument(
        '--actor_name',
        type=str,
        default=constants.DEFAULT_ACTOR_NAME,
        help='Actor/interviewee name.',
    )
    parser.add_argument(
        '--audience_name',
        type=str,
        default=constants.DEFAULT_AUDIENCE_NAME,
        help='Audience/interviewer name.',
    )
    parser.add_argument(
        '--llm_type',
        type=str,
        default='openai',
        choices=['openai', 'local'],
        help='LLM type: openai or local.',
    )
    parser.add_argument(
        '--local_model',
        type=str,
        default='llama3.1:8b',
        help='Local model name (for Ollama).',
    )
    parser.add_argument(
        '--pretty_trace',
        action='store_true',
        help='Print pretty conversation trace (default: False).',
    )
    parser.add_argument(
        '--no_plots',
        action='store_true',
        help='Disable plotting (default: plots are generated if matplotlib is available).',
    )
    parser.add_argument(
        '--enable_info_flow_logging',
        action='store_true',
        help='Enable information flow history logging (captures all LLM prompts and responses).',
    )
    parser.add_argument(
        '--enable_simplified_log',
        action='store_true',
        help='Enable simplified information flow log (human-readable format). Requires --enable_info_flow_logging.',
    )
    parser.add_argument(
        '--simplified_log_format',
        type=str,
        default='compact',
        choices=['compact', 'markdown', 'text'],
        help='Format for simplified log: compact (one line per interaction), markdown (structured), or text (indented).',
    )
    parser.add_argument(
        '--save_component_logs',
        action='store_true',
        help='Save Concordia component-level logs to JSON file (component state and behavior logs).',
    )
    parser.add_argument(
        '--enable_self_assessment',
        action='store_true',
        help='Enable self-assessment component (ensures responses align with traits, norms, and goals).',
    )
    parser.add_argument(
        '--consistency_threshold',
        type=float,
        default=0.7,
        help='Minimum consistency score (0-1) to accept response without revision (default: 0.7).',
    )
    parser.add_argument(
        '--disable_revision',
        action='store_true',
        help='Disable revision of inconsistent responses (only log assessments).',
    )
    parser.add_argument(
        '--no_instructions',
        action='store_true',
        help='Disable Instructions component (role-playing context).',
    )
    parser.add_argument(
        '--no_self_perception',
        action='store_true',
        help='Disable SelfPerception component ("who am I?" questions).',
    )
    parser.add_argument(
        '--enable_situation_perception',
        action='store_true',
        help='Enable SituationPerception component ("what situation am I in?" questions).',
    )
    parser.add_argument(
        '--enable_person_by_situation',
        action='store_true',
        help='Enable PersonBySituation component ("what would I do?" reasoning). Requires --enable_situation_perception.',
    )
    parser.add_argument(
        '--no_world_building',
        action='store_true',
        help='Disable 2A25 world-building context (Cadens, Riffers narrative).',
    )
    parser.add_argument(
        '--no_interview_context',
        action='store_true',
        help='Disable interview-specific context in world-building.',
    )
    parser.add_argument(
        '--use_trait_paragraph',
        action='store_true',
        help='Use one LLM-generated paragraph per agent for personality (instead of score-based traits). Adds 1 LLM call per agent.',
    )
    parser.add_argument(
        '--interview_role_preset',
        type=str,
        default='product_manager',
        help='Interview role preset: product_manager, customer_service. Sets role text and optional question/experience banks.',
    )
    parser.add_argument(
        '--no_question_bank',
        action='store_true',
        help='Do not append question bank to interviewer (audience) context.',
    )
    parser.add_argument(
        '--no_experience_bank',
        action='store_true',
        help='Do not append experience bank to interviewee (actor) context.',
    )
    parser.add_argument(
        '--actor_has_norms',
        action='store_true',
        help='Give the interviewee (actor) the same cultural norms as the interviewer (audience).',
    )
    parser.add_argument(
        '--use_option_space',
        action='store_true',
        help='[Experimental] Generate 4 response options per turn then choose one (2 LLM calls per turn for actor and audience).',
    )
    parser.add_argument(
        '--enable_question_checks',
        action='store_true',
        help='After the run, ask the model to summarize situation and personality per agent (2 LLM calls per agent, for analysis/debugging).',
    )
    parser.add_argument(
        '--no_full_2a25',
        action='store_true',
        help='Use minimal generic world-building text instead of full 2A25/Cadens/Riffers narrative.',
    )
    parser.add_argument(
        '--use_memory_check',
        action='store_true',
        help='Inject LLM-generated full-conversation summary into audience and actor prompts (1 extra LLM call per turn).',
    )
    args = parser.parse_args()

    # Validate consistency_threshold
    if not 0.0 <= args.consistency_threshold <= 1.0:
        parser.error("--consistency_threshold must be between 0.0 and 1.0")

    # Validate: simplified log requires info flow logging
    if args.enable_simplified_log and not args.enable_info_flow_logging:
        parser.error("--enable_simplified_log requires --enable_info_flow_logging")

    # Create output directory
    save_dir = utils.create_output_directory(args.save_dir)

    return ConversationConfig(
        turns=args.turns,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        window=args.window,
        outfile=args.outfile,
        no_audience_norms=args.no_audience_norms,
        no_traits=args.no_traits,
        no_context=args.no_context,
        seed=args.seed,
        save_dir=save_dir,
        actor_name=args.actor_name,
        audience_name=args.audience_name,
        llm_type=args.llm_type,
        local_model=args.local_model,
        print_trace=args.pretty_trace,
        no_plots=args.no_plots,
        enable_info_flow_logging=args.enable_info_flow_logging,
        enable_simplified_log=args.enable_simplified_log,
        simplified_log_format=args.simplified_log_format,
        save_component_logs=args.save_component_logs,
        enable_self_assessment=args.enable_self_assessment,
        consistency_threshold=args.consistency_threshold,
        disable_revision=args.disable_revision,
        no_instructions=args.no_instructions,
        no_self_perception=args.no_self_perception,
        enable_situation_perception=args.enable_situation_perception,
        enable_person_by_situation=args.enable_person_by_situation,
        no_world_building=args.no_world_building,
        no_interview_context=args.no_interview_context,
        traits_file=args.traits_file,
        use_trait_paragraph=args.use_trait_paragraph,
        interview_role_preset=args.interview_role_preset,
        no_question_bank=args.no_question_bank,
        no_experience_bank=args.no_experience_bank,
        actor_has_norms=args.actor_has_norms,
        use_option_space=args.use_option_space,
        enable_question_checks=args.enable_question_checks,
        use_full_2a25_world=not args.no_full_2a25,
        use_memory_check=args.use_memory_check,
    )


def validate_api_key(config: ConversationConfig) -> str:
    """Validate and return API key for OpenAI."""
    if config.llm_type != 'openai':
        return ''

    api_key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not api_key:
        print(
            '\nERROR: OPENAI_API_KEY environment variable required for OpenAI.\n',
            file=sys.stderr,
        )
        print('To set the API key:', file=sys.stderr)
        print('  PowerShell: $Env:OPENAI_API_KEY = "sk-your-api-key-here"', file=sys.stderr)
        print('  CMD:        set OPENAI_API_KEY=sk-your-api-key-here', file=sys.stderr)
        print('  Or create: projects/impression_management_standard/.env with: OPENAI_API_KEY=sk-your-api-key-here\n', file=sys.stderr)
        print('Alternatively, use a local model:', file=sys.stderr)
        print('  python projects/impression_management_standard/main.py --turns 2 --llm_type local\n', file=sys.stderr)
        sys.exit(1)

    return api_key
