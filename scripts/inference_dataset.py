from argparse import ArgumentParser
import logging

from datasets import Dataset as HuggingFaceDataset
from datasets import load_from_disk

from src.models.openbiollm import OpenBioLLM
from src.models.prometheus import Prometheus
from src.pipelines.dataset_inference_pipeline import HFModelDatasetInferencePipeline, ModelDatasetInferencePipeline
from src.data.utils import rows_to_chat, rows_to_chat_multi

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    force=True # This ensures we override any existing logger configuration
)

parser = ArgumentParser(description='Runs inference on a dataset partition.')

parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
parser.add_argument('--tokenizer', type=str, required=False, default=None, help='Tokenizer checkpoint')
parser.add_argument('--dataset', type=str, required=True, help='Path to dataset (csv or huggingface on disk)')
parser.add_argument('--output_path', type=str, required=True, help='Path where the output dataset will be saved')
parser.add_argument('--max_rows_to_process', type=int, default=None, help='Maximum number of rows to process')
parser.add_argument('--rows_to_chat', type=str, default='none', choices=['none', 'single', 'multi'], help='Chat format: none (no chat), single (single few-shot), multi (multiple few-shot)')
parser.add_argument('--input_column', type=str, default='PROMPT', help='Column to use as input')
parser.add_argument('--output_column', type=str, default='OUTPUT', help='Column to use as output')
parser.add_argument('--apply_chat_template', type=str, default='true', choices=['true', 'false'], help='Whether to apply the chat template or not (true/false)')
parser.add_argument('--hf', type=str, default='false', required=False, choices=['true', 'false'], help='Whether to use HuggingFace for inference (true/false)')
parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size to use (if using Huggingface for inference)')
parser.add_argument('--system_prompt', type=str, default='none', required=False, help='System prompt to use (none, bio, custom). If custom, use the format custom:<the prompt you want>')

SYSTEM_PROMPTS = {
    'none': None,
    'bio': OpenBioLLM.SYSTEM_PROMPT,
}

def get_system_prompt(prompt_type: str):

    if prompt_type.startswith('custom:'):
        return prompt_type.replace('custom:', '')

    return SYSTEM_PROMPTS[prompt_type]


def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() == 'true'


def get_rows_to_chat_function(chat_type: str):
    """Get the appropriate rows_to_chat function based on type"""
    if chat_type == 'single':
        return rows_to_chat
    elif chat_type == 'multi':
        return rows_to_chat_multi
    else:  # 'none'
        return None
    

def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    system_prompt = get_system_prompt(args.system_prompt)
    
    # Get the appropriate rows_to_chat function
    rows_to_chat_func = get_rows_to_chat_function(args.rows_to_chat)
    apply_chat_template = str_to_bool(args.apply_chat_template)
    use_hf = str_to_bool(args.hf)

    pipeline_args = {}
    if use_hf:
        pipeline = HFModelDatasetInferencePipeline(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )
        pipeline_args['batch_size'] = args.batch_size
    else:
        pipeline = ModelDatasetInferencePipeline(
            model_path=args.checkpoint,
            tokenizer_path=args.tokenizer
        )

    dataset_path: str = args.dataset

    if dataset_path.endswith('.csv'):
        dataset = HuggingFaceDataset.from_csv(args.dataset)
    else:
        dataset = load_from_disk(dataset_path)

    output_dataset = pipeline(
        dataset, 
        input_column=args.input_column, 
        output_column=args.output_column,
        max_rows_to_process=args.max_rows_to_process,
        rows_to_chat=rows_to_chat_func,
        apply_chat_template=apply_chat_template,
        system_prompt=system_prompt,
        **pipeline_args
    )

    output_dataset.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()
