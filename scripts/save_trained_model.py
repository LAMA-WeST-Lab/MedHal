import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # Load tokenizer from base model. The adapter directory has a tokenizer saved
    # with an unrecognized backend class (TokenizersBackend), so we load from base
    # and add the pad token that was added during training to match vocab size 128257.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=True, trust_remote_code=True)
    # During training, run_sft.py added '<finetune-pad-token>' because Llama 3's
    # pad_token_id == eos_token_id, bumping vocab from 128256 → 128257.
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<finetune-pad-token>"})

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    print("BASE MODEL LOADED")
    print(f"Base model vocab size:  {model.config.vocab_size}")
    print(f"Trained tokenizer size: {len(tokenizer)}")

    # Resize embeddings to match the tokenizer vocab size used during training.
    # Training may have added tokens (e.g. special tokens), causing a size mismatch
    # between the adapter checkpoint (128257) and the base model (128256).
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing token embeddings: {model.config.vocab_size} → {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Load adapter
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("ADAPTERS LOADED")

    # Merge LoRA weights into base model
    model = model.merge_and_unload()

    print("MODEL MERGED")

    # Save merged model
    model.save_pretrained(args.output_dir)

    print('SAVED MERGED MODEL')

    # Save tokenizer (important!)
    tokenizer.save_pretrained(args.output_dir)

    print("SAVED TOKENIZER")


if __name__ == "__main__":
    main()