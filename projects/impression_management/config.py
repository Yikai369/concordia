"""Configuration parsing and validation."""

import argparse
import os
import sys

from projects.impression_management import constants
from projects.impression_management import utils
from projects.impression_management.models import ConversationConfig

AUDIENCE_TRAITS_SPREADSHEET = (
    r'C:\Users\Jennifer Zhang\concordia\projects\impression_management\data\traits\autism-measures-compilation.xlsx'
)


def parse_arguments() -> ConversationConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Impression Management PE conversation (Concordia framework).'
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
        default='gpt-4o',
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
        '--trait_mode',
        type=str,
        default=constants.DEFAULT_TRAIT_MODE,
        choices=constants.TRAIT_MODE_CHOICES,
        help=(
            'Trait initialization mode: '
            'none (no traits for either agent), '
            'actor_only (only actor has traits paragraph).'
        ),
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
        '--use_memory_check',
        action='store_true',
        help='Inject LLM-generated full-conversation summary into prompts (1 extra LLM call per turn).',
    )
    parser.add_argument(
        '--enable_self_assessment',
        action='store_true',
        help='Wrap actor act component with self-assessment and optional revision.',
    )
    parser.add_argument(
        '--consistency_threshold',
        type=float,
        default=0.7,
        help='Minimum consistency score (0-1) to accept actor response without revision.',
    )
    parser.add_argument(
        '--disable_revision',
        action='store_true',
        help='Disable revision of inconsistent actor responses (only log assessments).',
    )
    parser.add_argument(
        '--save_component_logs',
        action='store_true',
        help='Save component-level logs to component_logs.json in output directory.',
    )
    args = parser.parse_args()

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
        trait_mode=args.trait_mode,
        no_context=args.no_context,
        seed=args.seed,
        save_dir=save_dir,
        actor_name=args.actor_name,
        audience_name=args.audience_name,
        audience_traits_spreadsheet=AUDIENCE_TRAITS_SPREADSHEET,
        llm_type=args.llm_type,
        local_model=args.local_model,
        use_memory_check=args.use_memory_check,
        enable_self_assessment=args.enable_self_assessment,
        consistency_threshold=args.consistency_threshold,
        disable_revision=args.disable_revision,
        save_component_logs=args.save_component_logs,
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
        print('  Or create: projects/impression_management/.env with: OPENAI_API_KEY=sk-your-api-key-here\n', file=sys.stderr)
        print('Alternatively, use a local model:', file=sys.stderr)
        print('  python projects/impression_management/main.py --turns 2 --llm_type local\n', file=sys.stderr)
        sys.exit(1)

    return api_key
