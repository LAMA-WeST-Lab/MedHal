# MedHal

Code for [MedHal: An Evaluation Dataset for Medical Hallucination Detection](https://arxiv.org/abs/2504.08596): training and evaluating models that judge whether a medical statement is factually supported by a context.

## Datasets

Source datasets: MedQA, MedMCQA, SumPubMed, MedNLI, Augmented Clinical Notes.

Released MedHal datasets on Hugging Face:

- [GM07/medhal](https://huggingface.co/datasets/GM07/medhal): raw, unfiltered, unbalanced (~800k samples)
- [GM07/medhal-lf](https://huggingface.co/datasets/GM07/medhal-lf): length-filtered (context + statement < 30000 characters, fits an 8192-token window)
- [GM07/medhal-lf-bal](https://huggingface.co/datasets/GM07/medhal-lf-bal): length-filtered and task-balanced, with `train`/`val`/`test` splits (use this for training)

Each sample has: `id, context, statement, label (bool), explanation, inner_id, source, synthetic`. The explanation of a factual statement is `"The statement is factual."`.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

GPU inference uses [vLLM](https://github.com/vllm-project/vllm) and shards across all visible GPUs (`tensor_parallel_size = torch.cuda.device_count()`).

## Reproducing the dataset (optional)

```bash
# 1. Fetch source datasets to a local folder
python -m scripts.data.fetch_datasets --out <dir>

# 2. Generate generation prompts for one source dataset
python -m scripts.data.generate_med_hal_prompts \
  --dataset_path <path> \
  --dataset MedQA|MedMCQA|AugmentedClinicalNotes|MedNLI|SumPubmed \
  --out <prompts.csv>

# 3. Run LLM inference over the prompts
python -m scripts.inference_dataset \
  --checkpoint <model> \
  --dataset <prompts.csv> \
  --output_path <outputs.csv> \
  --input_column PROMPT --output_column OUTPUT

# 4. Validate statements (QA sources)
python -m scripts.data.validate_qa_statements --dataset <csv> --out <validated.csv>

# 5. Build canonical MedHal samples (see src/data/med_hal/medhal.py: MedHal.from_*)
```

`MedHal.from_medqa / from_medmcqa / from_augmented_clinical_notes` expect paired factual/non-factual rows kept adjacent and sorted — do not shuffle before mapping. `MedHal.from_all` shuffles once after concatenation.

## Training

```bash
cp configs/train_example.yaml my_train.yaml
# edit model_checkpoint, dataset_path, output_dir
python -m scripts.training.train_med_hal my_train.yaml
```

Variants: `scripts/training/train_med_hal_deepspeed.py` (DeepSpeed) and `scripts/training/train_umed_hal.py` (Unsloth), plus the lower-level SFT entrypoint `src/training/scripts/run_sft.py`. Training formats samples with `src/data/formatter.py` and computes loss only after the `### Factual` marker (`DataCollatorForCompletionOnlyLM`). Supported `model_wrapper` values: `medhal, halloumi, prometheus, openbiollm`.

## Evaluation

`scripts/medhal_eval.py` runs 0-shot inference; `scripts/medhal_eval_few_shots.py` runs few-shot inference (the first N rows of the `test` split are used as examples and excluded from scoring). The eval dataset on disk must be a `DatasetDict` with a `test` split:

```bash
python -c "
from datasets import Dataset as HuggingFaceDataset, DatasetDict
ds = HuggingFaceDataset.from_csv('medhal_test.csv')
DatasetDict({'test': ds}).save_to_disk('dataset_for_eval')
"
python -m scripts.medhal_eval \
  --dataset dataset_for_eval \
  --model <model> --tokenizer <model> \
  --model_type medhal --model_wrapper medhal \
  --temperature 0.0 --top_p 1.0 --top_k -1 --min_p 0.0 \
  --out outputs.csv
```

Score the outputs with `src/evaluation/medhal_parser.py` (`MedHalParser.evaluate`): accuracy/precision/recall/F1 over valid rows plus an `invalid` split for unparseable outputs; embedding similarity (cosine over `all-MiniLM-L6-v2` embeddings) is reported on correctly predicted non-factual rows only.

```python
from src.evaluation.medhal_parser import MedHalParser
results = MedHalParser().evaluate("outputs.csv", add_prompt=False)
print({k: v for k, v in results.items() if k in ("accuracy", "precision", "recall", "f1", "embsim")})
```

Notes:

- MedHal eval prompts are raw text (`apply_chat_template=False`). Few-shot defaults to raw-text concatenation (matches training); pass `--use_chat_template` only for base instruction-tuned models. `halloumi` uses its own structured-tag few-shot format.
- If the tokenizer has no chat template, set `MEDHAL_FALLBACK_CHAT_TEMPLATE` to a tokenizer to borrow one from (e.g. a Llama instruct model); otherwise chat-template inference raises an informative error.
- Prompts longer than the model's context window are skipped (`None` output), not truncated.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
