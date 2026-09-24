from typing import List, Optional
import logging
import inspect

import torch
from tqdm import tqdm
from vllm import LLM, RequestOutput, SamplingParams

import os

from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

from src.models.utils import load_model, load_tokenizer
from src.utils import batch_elements, run_inference

# Optional fallback chat template source, e.g. a local instruct-model tokenizer
# directory or HF model id whose `chat_template` should be reused when the
# active tokenizer does not define one. Example:
#   export MEDHAL_FALLBACK_CHAT_TEMPLATE="meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_CHAT_TEMPLATE = os.environ.get("MEDHAL_FALLBACK_CHAT_TEMPLATE")

def apply_chat_template(tokenizer, inputs):
    """
    Applies the chat template to the inputs
    """
    if not tokenizer.chat_template:
        if FALLBACK_CHAT_TEMPLATE is None:
            raise ValueError(
                "The tokenizer has no chat template. Pass a tokenizer with a "
                "chat template, set MEDHAL_FALLBACK_CHAT_TEMPLATE to a tokenizer "
                "to borrow one from, or run inference with apply_chat_template=False."
            )
        instruct_tok = AutoTokenizer.from_pretrained(FALLBACK_CHAT_TEMPLATE)
        tokenizer.chat_template = instruct_tok.chat_template

    return tokenizer.apply_chat_template(inputs, tokenize=False, add_generation_prompt=True)

class HFModelInferencePipeline:
    """
    Helper class to load HuggingFace models and perform inference using transformers
    """
    def __init__(self, model_path: str, tokenizer_path: str = None):

        if tokenizer_path is None:
            tokenizer_path = model_path

        self.model = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)

    def run_inference(self, inputs: List[str], max_new_tokens: int = 4096, batch_size: int = 1):
        """
        Runs inference of a model on a set of inputs

        Args:
            inputs: Inputs to run inference on
            batch_size: Number of inputs to run inference on at once
            max_new_tokens: Number of tokens to generated
        """

        batched_prompts = batch_elements(inputs, batch_size)
        results = []
        for batch in tqdm(batched_prompts, desc='Running inference', total=len(batched_prompts)):
            encodeds = self.tokenizer(batch, return_tensors="pt", padding=True)
            model_inputs = encodeds.to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_new_tokens)
            decoded = self.tokenizer.batch_decode(generated_ids)
            results.extend(decoded)

        return results

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)

class ModelInferencePipeline:

    def __init__(self, model_path: str, tokenizer_path: str = None):
        self.nb_gpus = torch.cuda.device_count()
        logger.info(f'Using {self.nb_gpus} GPUs')

        if tokenizer_path is None:
            tokenizer_path = model_path

        self.llm = LLM(
            model=model_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=self.nb_gpus,
            dtype='bfloat16',
        )
        self.tokenizer = load_tokenizer(tokenizer_path)


    def run_inference(self, inputs: List, max_new_tokens: int = 4096, verify_lengths: bool = True, sampling_params: dict = None):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            max_new_tokens: Maximum number of new tokens to generate
            verify_lengths: Whether to verify if prompts lengths are less than the max model length (creates errors with vLLM if that's the case, but adds overhead)
            sampling_params: Optional dictionary of sampling parameters passed to vLLM SamplingParams
        """

        logger.debug("First inference input: %s", inputs[0][:500] if inputs else None)

        sampling_kwargs = {
            'max_tokens': max_new_tokens,
            'temperature': 0.0,
            'top_k': -1,
        }
        if sampling_params is not None:
            for key, value in sampling_params.items():
                if value is not None:
                    sampling_kwargs[key] = value

        valid_sampling_keys = set(inspect.signature(SamplingParams).parameters.keys())
        filtered_sampling_kwargs = {
            key: value for key, value in sampling_kwargs.items() if key in valid_sampling_keys
        }

        ignored_sampling_keys = [
            key for key in sampling_kwargs.keys() if key not in valid_sampling_keys
        ]
        if ignored_sampling_keys:
            logger.warning(
                f'Ignoring unsupported sampling params for current vLLM version: {ignored_sampling_keys}'
            )

        params = SamplingParams(**filtered_sampling_kwargs)
        if not verify_lengths:

            outputs = self.llm.generate(inputs, sampling_params=params)

            return [output.outputs[0].text for output in outputs]

        # Get the maximum model length from the vLLM engine's config
        # This max_model_len is the total sequence length (prompt + generated tokens)
        # The check vLLM performs is on the prompt token IDs length itself.
        engine_model_config = self.llm.llm_engine.model_config
        vllm_max_model_len = engine_model_config.max_model_len

        results: List[Optional[str]] = [None] * len(inputs)

        valid_prompts: List[str] = []
        valid_indices: List[int] = []
        
        # Pre-filter prompts
        for i, prompt_text in enumerate(inputs):
            prompt_token_ids = self.tokenizer.encode(prompt_text)
            
            if len(prompt_token_ids) <= vllm_max_model_len - 1:
                valid_prompts.append(prompt_text)
                valid_indices.append(i)
            else:
                logger.warning(
                    f"Prompt at index {i} (token length {len(prompt_token_ids)}) "
                    f"exceeds vLLM max model length ({vllm_max_model_len}). Skipping."
                )

        if not valid_prompts:
            logger.info("No valid prompts to process after length check.")
            return results
        
        try:
            # Generate outputs only for valid prompts
            # vLLM's generate can handle a list of prompts
            generated_outputs: List[RequestOutput] = self.llm.generate(valid_prompts, sampling_params=params)
            
            # Populate results for valid prompts at their original positions
            for i, output_obj in enumerate(generated_outputs):
                original_index = valid_indices[i]
                if output_obj.outputs: # Check if there are any output sequences
                    results[original_index] = output_obj.outputs[0].text.strip()
                else:
                    logger.warning(f"No output generated for prompt at original index {original_index} (valid prompt: '{valid_prompts[i][:50]}...').")
                    results[original_index] = None # Explicitly set to None if no output
        except Exception as e:
            logger.error(f"Error during vLLM generation: {e}")
            for original_idx in valid_indices:
                results[original_idx] = None

        return results

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)

class ClassifierModelInferencePipeline:

    def __init__(self, model_path: str, tokenizer_path: str = None):
        if tokenizer_path is None:
            tokenizer_path = model_path
        self.model = LLM(model=model_path, tokenizer=tokenizer_path)
        self.tokenizer = load_tokenizer(tokenizer_path)

    def run_inference(self, inputs: List, batch_size: int = 16, apply_chat_template: bool = False):
        """
        Runs inference on the inputs using vllm

        Args:
            inputs: List of inputs to run inference on
            batch_size: Number of inputs to process in parallel
        """
        predictions = []

        if apply_chat_template:
            inputs = self.apply_chat_template(inputs)

        batches = batch_elements(inputs, batch_size=batch_size)
        for batch in tqdm(batches, total=len(batches), desc='Processing batches'):
            outputs = self.model.classify(batch)
            predictions.extend([output.outputs.probs for output in outputs])

        return predictions

    def apply_chat_template(self, inputs: List):
        return apply_chat_template(self.tokenizer, inputs)
