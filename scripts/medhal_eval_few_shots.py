from argparse import ArgumentParser
import logging

from src.pipelines.dataset_inference_pipeline import ModelDatasetInferencePipeline

logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

from datasets import load_from_disk

from src.data.formatter import Formatter
from src.data.formatter import (
    Formatter,
    _build_halloumi_context_block,
    _build_halloumi_request_block,
    _build_halloumi_response_block,
    _build_halloumi_response_prefix,
)
from src.models.utils import load_tokenizer

parser = ArgumentParser(description='Program that evaluates a model on the medhal dataset with few-shot examples')

parser.add_argument('--dataset', type=str, required=True, help='Path to medhal dataset')
parser.add_argument('--model', type=str, required=True, help='Path to model')
parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
parser.add_argument('--model_type', type=str, required=True, help='Will indicate how the prompt will be formatted (medhal or general)')
parser.add_argument('--model_wrapper', type=str, default='medhal', help='Model wrapper type: medhal, halloumi, prometheus, or openbiollm (default: medhal)')
parser.add_argument('--out', type=str, required=True, help='Where the dataset will be saved')
parser.add_argument('--num_few_shots', type=int, default=3, help='Number of few-shot examples to use (default: 3)')
parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (default: 0.0)')
parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling top_p (default: 1.0)')
parser.add_argument('--top_k', type=int, default=-1, help='Top-k sampling (default: -1, disabled)')
parser.add_argument('--min_p', type=float, default=0.0, help='Minimum probability sampling threshold (default: 0.0)')
parser.add_argument('--use_chat_template', action='store_true', 
                    help='Use chat template format for few-shots. Default is raw text format (matching training). '
                         'Use this flag for base models that expect chat format. '
                         'For fine-tuned models trained without chat templates, omit this flag.')

def create_rows_to_chat_with_few_shots(few_shot_samples, formatter):
    """
    Creates a rows_to_chat function that includes few-shot examples
    
    Args:
        few_shot_samples: HuggingFace dataset with few-shot examples
        formatter: Formatter instance for formatting samples
    
    Returns:
        A function that converts rows to chat format with few-shots
    """
    def rows_to_chat_with_few_shots(rows):
        """Create chat format with few-shot examples as separate turns"""
        chats = []
        
        for i in range(len(rows['context'])):
            # Build conversation with task description once, then few-shot examples
            messages = []
            
            # Add task description as first user message
            task_description = """### Task Description
- You will evaluate whether a medical statement is factually accurate.
- The statement may reference a provided context.
- Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
- In order to answer YES, everything in the statement must be supported by the context.
- In order to answer NO, there must be at least one piece of information in the statement that is not supported by the context."""
            messages.append({'role': 'user', 'content': task_description})
            messages.append({'role': 'assistant', 'content': 'I understand. I will evaluate medical statements for factual accuracy based on the provided context.'})
            
            # Add few-shot examples
            for j in range(len(few_shot_samples)):
                # Create example question (without task description)
                example_question = formatter.format_fewshot_target(
                    context=few_shot_samples['context'][j],
                    statement=few_shot_samples['statement'][j]
                )
                messages.append({'role': 'user', 'content': example_question})
                
                # Create example answer
                example_label = few_shot_samples['label'][j]
                example_explanation = few_shot_samples['explanation'][j] if few_shot_samples['explanation'][j] else ''
                example_response = f"{'YES' if example_label else 'NO'}\n\n### Explanation\n{example_explanation}"
                messages.append({'role': 'assistant', 'content': example_response})
            
            # Add target sample (the one we want the model to answer)
            target_prompt = formatter.format_fewshot_target(
                context=rows['context'][i],
                statement=rows['statement'][i]
            )
            messages.append({'role': 'user', 'content': target_prompt})
            
            chats.append(messages)
        
        return chats
    
    return rows_to_chat_with_few_shots


def format_halloumi_few_shots_as_raw_text(few_shot_samples, formatter, rows):
    """
    Format few-shot examples in the HallOumi structured-tag format.
    Each example is a complete context+request+response block.
    The target gets a partial response block for the model to complete.
    """
    prompts = []

    for i in range(len(rows['context'])):
        parts = []

        for j in range(len(few_shot_samples)):
            example_label = few_shot_samples['label'][j]
            example_explanation = few_shot_samples['explanation'][j] or ''

            parts.append(
                _build_halloumi_context_block(few_shot_samples['context'][j])
                + _build_halloumi_request_block(few_shot_samples['statement'][j])
                + _build_halloumi_response_block(example_label, example_explanation)
            )

        parts.append(
            _build_halloumi_context_block(rows['context'][i])
            + _build_halloumi_request_block(rows['statement'][i])
            + _build_halloumi_response_prefix()
        )

        prompts.append(''.join(parts))

    return prompts


def format_few_shots_as_raw_text(few_shot_samples, formatter, rows):
    """
    Format few-shot examples as RAW TEXT (matching training format).
    Use this for fine-tuned models that were trained without chat templates.
    
    Args:
        few_shot_samples: HuggingFace dataset with few-shot examples
        formatter: Formatter instance for formatting samples
        rows: The batch of rows to process
    
    Returns:
        List of raw text prompts with few-shot examples concatenated
    """
    prompts = []
    
    task_description = """### Task Description
- You will evaluate whether a medical statement is factually accurate.
- The statement may reference a provided context.
- Respond with "YES" if the statement is factually correct or "NO" if it contains inaccuracies.
- In order to answer YES, everything in the statement must be supported by the context.
- In order to answer NO, there must be at least one piece of information in the statement that is not supported by the context.
"""
    
    for i in range(len(rows['context'])):
        # Start with task description
        prompt_parts = [task_description]
        
        # Add few-shot examples in the SAME format as training data
        for j in range(len(few_shot_samples)):
            example_label = few_shot_samples['label'][j]
            example_explanation = few_shot_samples['explanation'][j] if few_shot_samples['explanation'][j] else ''
            
            # Format example exactly like training data (but without task description to avoid repetition)
            example = formatter.format_fewshot_example(
                context=few_shot_samples['context'][j],
                statement=few_shot_samples['statement'][j],
                label=example_label,
                explanation=example_explanation
            )
            prompt_parts.append(example)
        
        # Add the target sample (inference format - no label/explanation)
        target = formatter.format_fewshot_target(
            context=rows['context'][i],
            statement=rows['statement'][i]
        )
        prompt_parts.append(target)
        
        prompts.append('\n'.join(prompt_parts))
    
    return prompts

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    assert args.model_type in ['medhal', 'general']
    assert args.num_few_shots >= 0, 'num_few_shots must be non-negative'
    assert args.model_wrapper in ['medhal', 'halloumi', 'prometheus', 'openbiollm'], f'model_wrapper must be one of: medhal, halloumi, prometheus, openbiollm'

    dataset = load_from_disk(args.dataset)
    test = dataset['test']

    print(f'Total samples: {len(test)}')

    sampling_params = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'min_p': args.min_p,
    }

    if args.model_type == 'medhal':
        tokenizer = load_tokenizer(args.tokenizer)
        formatter = Formatter(tokenizer, training=False, model_wrapper=args.model_wrapper)

        # Separate few-shot examples and test samples
        num_few_shots = min(args.num_few_shots, len(test))
        
        if num_few_shots > 0:
            print(f'Using {num_few_shots} few-shot examples')
            few_shot_samples = test.select(range(num_few_shots))
            test_samples = test.select(range(num_few_shots, len(test)))
            
            print(f'Processing {len(test_samples)} test samples with few-shot examples')

            if args.model_wrapper == 'halloumi':
                # halloumi has its own structured-tag few-shot format; chat template is not applicable
                print('Using HALLOUMI structured-tag format for few-shots')

                def format_batch_halloumi(rows):
                    return {'text': format_halloumi_few_shots_as_raw_text(few_shot_samples, formatter, rows)}

                test_samples = test_samples.map(
                    format_batch_halloumi,
                    batched=True,
                    desc='Formatting dataset (halloumi few-shot)'
                )

                print('Sample few-shot + test prompt (halloumi):')
                print(test_samples[0]['text'][:1500] + '...\n')

                test = test_samples

                pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
                pipeline(
                    test,
                    apply_chat_template=False,
                    saving_path=args.out,
                    input_column='text',
                    sampling_params=sampling_params
                )
            
            elif args.use_chat_template:
                # Use chat template format (for base instruction-tuned models)
                print('Using CHAT TEMPLATE format for few-shots (--use_chat_template flag)')
                rows_to_chat_fn = create_rows_to_chat_with_few_shots(few_shot_samples, formatter)
                
                # Show a sample
                print('Sample few-shot + test prompt (chat format):')
                sample_prompts = rows_to_chat_fn({
                    'context': [test_samples[0]['context']],
                    'statement': [test_samples[0]['statement']]
                })
                print(sample_prompts[0][0]['content'])
                
                test = test_samples
                
                # Run inference with chat template
                pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
                pipeline(
                    test,
                    rows_to_chat=rows_to_chat_fn,
                    apply_chat_template=False,  # rows_to_chat already returns chat format
                    saving_path=args.out,
                    input_column='context',  # Not used when rows_to_chat is provided
                    sampling_params=sampling_params
                )
            else:
                # Use raw text format (for fine-tuned models trained without chat templates)
                print('Using RAW TEXT format for few-shots (matching training format)')
                print('NOTE: Use --use_chat_template for base instruction-tuned models')
                
                # Create a function that formats rows as raw text with few-shots
                def format_batch_with_few_shots(rows):
                    return {'text': format_few_shots_as_raw_text(few_shot_samples, formatter, rows)}
                
                test_samples = test_samples.map(
                    format_batch_with_few_shots,
                    batched=True,
                    desc='Formatting dataset with few-shot examples'
                )
                
                # Show a sample
                print('Sample few-shot + test prompt (raw text):')
                print(test_samples[0]['text'][:1500] + '...\n')
                
                test = test_samples
                
                # Run inference WITHOUT chat template (raw text)
                pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
                pipeline(
                    test,
                    apply_chat_template=False,
                    saving_path=args.out,
                    input_column='text',
                    sampling_params=sampling_params
                )
        else:
            # num_few_shots == 0: Standard inference without few-shots
            print('No few-shot examples specified (num_few_shots=0)')
            test = test.map(
                lambda x: {'text': formatter.format_sample(x['context'], x['statement'], None, None)},
                desc='Formatting dataset'
            )
            print('Sample prompt:')
            print(test[0]['text'][:500] + '...\n')
            
            pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
            pipeline(
                test,
                apply_chat_template=False,
                saving_path=args.out,
                input_column='text',
                sampling_params=sampling_params
            )

if __name__ == '__main__':
    main()
