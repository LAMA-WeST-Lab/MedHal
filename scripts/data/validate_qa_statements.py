from argparse import ArgumentParser
import logging
import os
import json

from src.data.qa_validation.qa_validator import QAValidator

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True  # This ensures we override any existing logger configuration
)

logger = logging.getLogger(__name__)

parser = ArgumentParser(
    description='Program that generates validation prompts for medical statements using QAValidator'
)
parser.add_argument(
    '--dataset',
    type=str,
    required=True,
    help='Path to dataset containing medical statements (CSV or HuggingFace on disk)'
)
parser.add_argument(
    '--out',
    type=str,
    required=True,
    help='Output path where the validation prompts will be saved (CSV file)'
)
parser.add_argument(
    '--statement_column',
    type=str,
    default='statement',
    help='Name of the column containing statements (default: statement)'
)
parser.add_argument(
    '--few_shot_examples',
    type=str,
    default=None,
    help='Path to JSON file containing few-shot examples. Each example should have: statement, answer, and optionally context'
)
parser.add_argument(
    '--output_format',
    type=str,
    default='simple',
    choices=['simple', 'chat'],
    help='Output format: simple (text prompts) or chat (multi-turn chat format for inference pipeline)'
)

parser.add_argument(
    '--delimiter',
    type=str,
    default=None,
    help='CSV delimiter (comma, tab, semicolon, etc.). If not specified, will auto-detect.'
)


def load_few_shot_examples(file_path: str):
    """Load few-shot examples from JSON file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Few-shot examples file {file_path} does not exist")
    
    with open(file_path, 'r') as f:
        examples = json.load(f)
    
    logger.info(f'Loaded {len(examples)} few-shot examples from {file_path}')
    return examples


def main():
    args = parser.parse_args()

    print('Script called with args:', args)

    # Validate input
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset path {args.dataset} does not exist")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.out)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f'Created output directory: {output_dir}')

    # Load few-shot examples if provided
    few_shot_examples = None
    if args.few_shot_examples:
        few_shot_examples = load_few_shot_examples(args.few_shot_examples)

    # Initialize validator
    logger.info(f'Loading dataset from {args.dataset}')
    validator = QAValidator(args.dataset, few_shot_examples=few_shot_examples)

    # Generate validation prompts based on output format
    if args.output_format == 'chat':
        logger.info('Generating chat-formatted validation data with few-shot examples')
        output_dataset = validator.generate_chat_data_for_inference()
    else:
        logger.info('Generating simple text validation prompts')
        output_dataset = validator.generate_validation_prompts()

    # Save to CSV
    logger.info(f'Saving validation prompts to {args.out}')
    output_dataset.to_csv(args.out, index=False)

    logger.info(f'Successfully generated {len(output_dataset)} validation prompts')
    if args.output_format == 'chat':
        print(f'Sample data:')
        print(f'  system_prompt: {output_dataset[0]["system_prompt"][:80]}...')
        print(f'  few_shot_messages: {output_dataset[0]["few_shot_messages"][:80]}...')
        print(f'  user_input: {output_dataset[0]["user_input"][:80]}...\n')
    else:
        print(f'Sample prompt:\n{output_dataset[0]["prompt"]}\n')


if __name__ == '__main__':
    main()
