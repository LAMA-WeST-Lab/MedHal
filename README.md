# MedHal

Code for [MedHal: A Synthetic Dataset for Medical Hallucination Detection](https://arxiv.org/abs/2504.08596): training and evaluating models that judge whether a medical statement is factually supported by a context.

## Datasets

Source datasets: MedQA, MedMCQA, SumPubMed, MedNLI, Augmented Clinical Notes.

Released MedHal datasets on Hugging Face:

- [LAMA-WeST/medhal-raw](https://huggingface.co/datasets/LAMA-WeST/medhal-raw): raw, unfiltered, unbalanced (~800k samples)
- [LAMA-WeST/medhal-lf_qafil_bal](https://huggingface.co/datasets/LAMA-WeST/medhal-lf_qafil_bal): length-filtered, qa-filtered and task-balanced, with `train`/`val`/`test` splits (use this for training)

## Trained Models

Released MedHal trained models on Hugging Face:

- [medhal-llama3-8b](https://huggingface.co/LAMA-WeST/medhal-llama3-8b)
- [medhal-openbiollm-8b](https://huggingface.co/LAMA-WeST/medhal-openbiollm-8b)
- [medhal-medllama3-8b](https://huggingface.co/LAMA-WeST/medhal-medllama3-8b)
- [medhal-halloumi-8b](https://huggingface.co/LAMA-WeST/medhal-halloumi-8b)
- [medhal-qwen3-8b](https://huggingface.co/LAMA-WeST/medhal-qwen3-8b)

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:.
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

GPU inference uses [vLLM](https://github.com/vllm-project/vllm).

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

## Training

```bash
cp configs/train_example.yaml my_train.yaml
# edit model_checkpoint, dataset_path, output_dir
python -m scripts.training.train_med_hal my_train.yaml
```

## Evaluation

`scripts/medhal_eval.py` runs 0-shot inference
`scripts/medhal_eval_few_shots.py` runs few-shot inference. 

Score the outputs with `src/evaluation/medhal_parser.py`

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
