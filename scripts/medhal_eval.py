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
from src.models.utils import load_tokenizer

parser = ArgumentParser(description='Program that evaluates a model on the medhal dataset')

parser.add_argument('--dataset', type=str, required=True, help='Path to medhal dataset')
parser.add_argument('--model', type=str, required=True, help='Path to model')
parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer')
parser.add_argument('--model_type', type=str, required=True, help='Will indicate how the prompt will be formatted (medhal or general)')
parser.add_argument('--model_wrapper', type=str, default='medhal', help='Model wrapper type: medhal, halloumi, prometheus, or openbiollm (default: medhal)')
parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature (default: 0.0)')
parser.add_argument('--top_p', type=float, default=1.0, help='Nucleus sampling top_p (default: 1.0)')
parser.add_argument('--top_k', type=int, default=-1, help='Top-k sampling (default: -1, disabled)')
parser.add_argument('--min_p', type=float, default=0.0, help='Minimum probability sampling threshold (default: 0.0)')
parser.add_argument('--out', type=str, required=True, help='Where the dataset will be saved')

VALID_MODEL_WRAPPERS = ['medhal', 'halloumi', 'prometheus', 'openbiollm']


def prompt_medhal(x, formatter):
    return formatter(x)


def main():

    args = parser.parse_args()

    print('Called with arguments : ', args)

    assert args.model_type in ['medhal', 'general']
    assert args.model_wrapper in VALID_MODEL_WRAPPERS, \
        f'model_wrapper must be one of: {", ".join(VALID_MODEL_WRAPPERS)}'

    dataset = load_from_disk(args.dataset)
    test = dataset['test']

    if args.model_type == 'medhal':
        tokenizer = load_tokenizer(args.tokenizer)
        formatter = Formatter(tokenizer, training=False, model_wrapper=args.model_wrapper)

        test = test.map(
            prompt_medhal,
            fn_kwargs={'formatter': formatter},
            desc='Formatting dataset'
        )

    print('SAMPLE RAW PROMPT:')
    print(test[0]['text'])
    print('=' * 100)

    pipeline = ModelDatasetInferencePipeline(args.model, args.tokenizer)
    pipeline(
        test,
        apply_chat_template=False,
        saving_path=args.out,
        input_column='text',
        sampling_params={
            'temperature': args.temperature,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'min_p': args.min_p,
        }
    )


if __name__ == '__main__':
    main()