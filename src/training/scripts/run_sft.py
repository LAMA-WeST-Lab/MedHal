#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified for the training of MedHal-models
"""
Supervised fine-tuning script for decoder language models.
"""

import os
import logging
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import datasets
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig, DataCollatorForLanguageModeling, set_seed, Mxfp4Config
from trl import SFTTrainer, SFTConfig

from src.training.utils import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    # SFTConfig,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)

from src.data.formatter import Formatter

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """
    Data collator used for completion tasks. It ensures that all the tokens of the labels are set to an 'ignore_index'
    when they do not come from the assistant. This ensure that the loss is only
    calculated on the completion made by the assistant.

    Args:
        response_template (`Union[str, List[int]]`): the template form that indicates the start of the response, typically something like
            '### Response:\n'. It can also be passed as tokenized ids, which can be useful when using a tokenizer that encodes the response
            differently if it does not have proper context.
        instruction_template (`Union[str, List[int]]`): the template form that indicates the start of the human instruction, typically something like
            '### Human:\n'. Useful for assistant-style conversation datasets. It can also be passed as tokenized ids.
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class. Note that this option currently has no effect but is present
             for flexibility and backwards-compatibility.
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens with
    """

    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Optional[Union[str, List[int]]] = None,
        *args,
        mlm: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)

        self.instruction_template = instruction_template
        if isinstance(instruction_template, str):
            # The user provides a string, must tokenize
            self.instruction_token_ids = self.tokenizer.encode(
                self.instruction_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.instruction_token_ids = instruction_template

        self.response_template = response_template
        if isinstance(response_template, str):
            # The user provides a string, must tokenize
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            # The user already provides the token ids
            self.response_token_ids = response_template

        if (
            not self.mlm
            and self.instruction_template
            and self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        ):
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )

        self.ignore_index = ignore_index

    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        if self.instruction_template is None:
            for i in range(len(examples)):
                response_token_ids_start_idx = None

                for idx in np.where(batch["labels"][i] == self.response_token_ids[0])[
                    0
                ]:
                    # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
                    if (
                        self.response_token_ids
                        == batch["labels"][i][
                            idx : idx + len(self.response_token_ids)
                        ].tolist()
                    ):
                        response_token_ids_start_idx = idx

                if response_token_ids_start_idx is None:
                    warnings.warn(
                        f"Could not find response key `{self.response_template}` in the "
                        f'following instance: {self.tokenizer.decode(batch["input_ids"][i])} '
                        f"This instance will be ignored in loss calculation. "
                        f"Note, if this happens often, consider increasing the `max_seq_length`."
                    )
                    batch["labels"][i, :] = self.ignore_index
                else:
                    response_token_ids_end_idx = response_token_ids_start_idx + len(
                        self.response_token_ids
                    )

                    # Make pytorch loss function ignore all tokens up through the end of the response key
                    batch["labels"][i, :response_token_ids_end_idx] = self.ignore_index

        return batch


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
    )

    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)
    column_names_val = list(raw_datasets["val"].features)

    intersection = set(column_names) & set(column_names_val)
    column_names = list(intersection)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    # https://medium.com/@parikshitsaikia1619/mistral-mastery-fine-tuning-fast-inference-guide-62e163198b06
    tokenizer.padding_side = "right"
    # tokenizer.pad_token = tokenizer.unk_token
    # tokenizer.pad_token_id = tokenizer.unk_token_id

    #####################
    # Apply MedHal template
    #####################
    formatter = Formatter(tokenizer, training=True, model_wrapper=data_args.model_wrapper)

    raw_datasets = raw_datasets.map(
        formatter,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying MedHal template",
    )

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["val"]

    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(raw_datasets["train"])), 1):
            logger.info(
                f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}"
            )

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    config_preview = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
    )

    quantization_config = None
    device_map = None

    if hasattr(config_preview, "quantization_config") and config_preview.quantization_config is not None:
        quant_method = getattr(config_preview.quantization_config, "quant_method", None)

        if quant_method and "mxfp4" in str(quant_method).lower():
            logger.info("MXFP4 detected → removing config quantization and enabling dequantization")

            # IMPORTANT: remove quantization from config
            config_preview.quantization_config = None

            # enable dequantization
            quantization_config = Mxfp4Config(dequantize=True)

    else:
        quantization_config = get_quantization_config(model_args)
        device_map = get_kbit_device_map() if quantization_config is not None else None


    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        config=config_preview,   # ← ADD THIS
    )
    
    # Handle use_cache separately for models that don't accept it as a kwarg
    use_cache_value = False if training_args.gradient_checkpointing else True

    MODELS_WITH_NESTED_CONFIG = {"qwen3_5"}
    MODELS_WITH_CONFIG_ONLY_USE_CACHE = {"gpt_oss"}

    if config_preview.model_type in MODELS_WITH_NESTED_CONFIG:
        config_preview.text_config.use_cache = use_cache_value
    else:
        # For all other models (including llama-based like HallOumi, OpenBioLLM),
        # set use_cache on the config directly rather than passing as a kwarg
        config_preview.use_cache = use_cache_value

    model_kwargs["config"] = config_preview

    # After building model_kwargs, override attn_implementation for GPT-OSS
    if config_preview.model_type == "gpt_oss":
        model_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs
    )

    # Check the first few parameters to verify dtype and gradient capability
    for name, param in model.named_parameters():
        if "layers.0.self_attn.q_proj" in name: # Check a standard layer
            logger.info(f"DEBUG: Parameter {name} dtype: {param.dtype}")
            logger.info(f"DEBUG: Parameter {name} requires_grad: {param.requires_grad}")
            break

    # Verify the model's overall float type
    model_dtype = next(model.parameters()).dtype
    logger.info(f"Overall model dtype: {model_dtype}")

    if model_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
        logger.error("CRITICAL: Model is not in a floating point format! Training will fail.")

    logger.info("*** Model loaded! ***")

    # validate_quantization_for_training() inside SFTTrainer checks model.config.quantization_config.
    # For gpt_oss, the custom config class re-attaches quantization_config after loading even when
    # dequantize=True. Since the model is already in bf16 at this point, strip it here.
    if hasattr(model.config, "quantization_config") and model.config.quantization_config is not None:
        quant_method = getattr(model.config.quantization_config, "quant_method", None)
        if quant_method and "mxfp4" in str(quant_method).lower():
            logger.info("Stripping residual MXFP4 quantization_config from model.config post-load")
            model.config.quantization_config = None

    # After model load and your existing strip block, add:
    if config_preview.model_type == "gpt_oss":
        # GptOssConfig reloads quantization_config from disk inside Trainer.__init__.
        # Override it at the class level so it always returns None regardless.
        _GptOssConfigClass = type(model.config)
        _GptOssConfigClass.quantization_config = property(
            fget=lambda self: None,
            fset=lambda self, v: None,  # silently swallow any re-assignment
        )
        logger.info("Patched GptOssConfig.quantization_config property to always return None")

    import json, tempfile, shutil

    if config_preview.model_type == "gpt_oss":
        # Write a clean config.json with quantization_config nulled out
        # so Trainer.__init__'s re-load from disk gets a clean copy
        clean_config_dir = tempfile.mkdtemp()
        shutil.copytree(model_args.model_name_or_path, clean_config_dir, dirs_exist_ok=True)
        
        config_path = os.path.join(clean_config_dir, "config.json")
        with open(config_path, "r") as f:
            config_json = json.load(f)
        config_json["quantization_config"] = None
        with open(config_path, "w") as f:
            json.dump(config_json, f, indent=2)
        
        # Point the model's config to this clean directory
        model.config.name_or_path = clean_config_dir
        logger.info(f"Wrote clean config.json (no quantization_config) to {clean_config_dir}")

    if tokenizer.pad_token_id is None:
        logger.warning("No pad token found, setting it to <finetune-pad-token>")
        tokenizer.add_special_tokens({'pad_token': '<finetune-pad-token>'})
        # TODO : What happens if the tokenizer is in a different folder, thus has the pad token id, but the model does not have the same length
        model.resize_token_embeddings(len(tokenizer)) 
        logger.warning(f"Pad token added")
    elif tokenizer.pad_token_id == tokenizer.eos_token_id:
        logger.warning("Pad token and eos token are the same, setting pad token to <finetune-pad-token>")
        tokenizer.add_special_tokens({'pad_token': '<finetune-pad-token>'})
        model.resize_token_embeddings(len(tokenizer)) 
        logger.warning(f"Pad token added")
    else:
        logger.info(f'Pad token is {tokenizer.pad_token}')

    assert (
        tokenizer.pad_token_id != tokenizer.eos_token_id
    ), "The tokenizer's pad token id and eos token id should not be the same."

    RESPONSE_TEMPLATE_MAP = {
        "medhal":     "### Factual",
        "halloumi":   "<|response|>",
        "prometheus": "...",
        "openbiollm": "### Evaluation",
    }
    response_template_context = RESPONSE_TEMPLATE_MAP[data_args.model_wrapper]

    example = train_dataset[0]["text"]
    assert response_template_context in example

    # Encode template in context to get correct token IDs (avoids BPE boundary issues)
    RESPONSE_TEMPLATE_PREFIX_MAP = {
        "medhal":     "### Statement\nsome statement\n\n",
        "halloumi":   "<end||request>",
        "prometheus": "\n\n",
        "openbiollm": "### Statement\nsome statement\n\n",
    }
    prefix_str = RESPONSE_TEMPLATE_PREFIX_MAP[data_args.model_wrapper]
    context_str = prefix_str + response_template_context

    full_ids = tokenizer.encode(context_str, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)

    # The response template IDs are whatever tokens remain after the prefix
    response_template_ids = full_ids[len(prefix_ids):]

    if data_args.model_wrapper == "halloumi":
        # Take a real example and check what token IDs are actually in it
        sample_text = train_dataset[0]["text"]
        sample_ids = tokenizer.encode(sample_text, add_special_tokens=True)

        logger.info(f"Response template IDs we're searching for: {response_template_ids}")

        # Find where <|response|> appears in the actual tokenized sample
        eval_str = "<|response|>"
        eval_char_pos = sample_text.find(eval_str)
        # Get a window of text around it and tokenize to see actual IDs
        window = sample_text[eval_char_pos - 20 : eval_char_pos + 20]
        window_ids = tokenizer.encode(window, add_special_tokens=False)
        logger.info(f"Token IDs around <|response|> in real sample: {window_ids}")
        logger.info(f"Decoded window: {repr(window)}")
        
        # Also search directly in sample_ids
        found = False
        for i in range(len(sample_ids) - len(response_template_ids) + 1):
            if sample_ids[i:i+len(response_template_ids)] == response_template_ids:
                found = True
                logger.info(f"Template found at position {i} in sample token IDs")
                break
        if not found:
            logger.warning("Template NOT found in sample — IDs mismatch!")
            # Show what IDs appear at the evaluation tag position
            # Tokenize prefix only to find approximate position
            pre_eval = sample_text[:eval_char_pos]
            pre_eval_ids = tokenizer.encode(pre_eval, add_special_tokens=True)
            pos = len(pre_eval_ids)
            logger.warning(f"IDs at/around evaluation position {pos}: {sample_ids[pos-2:pos+6]}")


    assert len(response_template_ids) > 0, (
        f"Could not extract response template IDs for wrapper '{data_args.model_wrapper}'. "
        f"Check that the prefix tokenizes cleanly and the template follows it."
    )

    logger.info(f"Response template: {repr(response_template_context)}")
    logger.info(f"Response template token IDs: {response_template_ids}")
    logger.info(f"Response template decoded: {tokenizer.decode(response_template_ids)}")

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
        mlm=False,
    )


    ########################
    # Initialize the Trainer
    ########################

    if data_args.model_wrapper == "halloumi":
        if not hasattr(training_args, "dataset_kwargs") or training_args.dataset_kwargs is None:
            training_args.dataset_kwargs = {}
        training_args.dataset_kwargs["skip_prepare_dataset"] = True
        
        # Pre-tokenize as fallback in case skip_prepare_dataset is not respected
        def tokenize_fn(examples):
            return tokenizer(examples["text"], truncation=True, max_length=training_args.max_seq_length)
        
        train_dataset = train_dataset.map(tokenize_fn, batched=True, num_proc=data_args.preprocessing_num_workers, desc="Pre-tokenizing train")
        eval_dataset = eval_dataset.map(tokenize_fn, batched=True, num_proc=data_args.preprocessing_num_workers, desc="Pre-tokenizing eval")

    import transformers.trainer_utils as _tutils

    _orig_validate = _tutils.validate_quantization_for_training

    def _patched_validate(model):
        # We know we've handled dequantization or stripping manually.
        # Do absolutely nothing and just return.
        logger.info("Forcing bypass of validate_quantization_for_training")
        return 

    _tutils.validate_quantization_for_training = _patched_validate

    # Also patch the reference already imported inside trainer.py
    import transformers.trainer as _trainer_mod
    _trainer_mod.validate_quantization_for_training = _patched_validate

    qc = getattr(model.config, "quantization_config", None)
    logger.info(f"DEBUG quantization_config type: {type(qc)}")
    logger.info(f"DEBUG quantization_config value: {qc}")
    if qc is not None:
        logger.info(f"DEBUG quant_method: {getattr(qc, 'quant_method', 'MISSING')}")
        logger.info(f"DEBUG quant_method str: {str(getattr(qc, 'quant_method', ''))}")
        logger.info(f"DEBUG dir(qc): {[x for x in dir(qc) if not x.startswith('__')]}")

    tc = getattr(model.config, "text_config", None)
    if tc:
        logger.info(f"DEBUG text_config.quantization_config: {getattr(tc, 'quantization_config', 'N/A')}")

    

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # Off tokenizer and dataset_text_field when using Data Collator
        # tokenizer=tokenizer,
        data_collator=data_collator,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##########
    # Evaluate
    ##########
    if True:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        try:
            trainer.create_model_card(**kwargs)
        except TypeError:
            trainer.create_model_card(tags=["alignment-handbook"])
            # Manually add finetuned_from info to the model card
            model_card_path = os.path.join(training_args.output_dir, "README.md")
            with open(model_card_path, "a") as f:
                f.write(f"\n## Training Details\n")
                f.write(f"- **Finetuned from:** {model_args.model_name_or_path}\n")
                f.write(f"- **Tags:** alignment-handbook\n")
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
