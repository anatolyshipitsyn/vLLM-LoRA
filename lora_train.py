import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel

def load_training_data(data_path: str) -> list:
    """ Load training data from JSON and format it for fine-tuning. """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_examples = []
    for project in data.get("projects", []):
        if not project.get("description"):
            continue

        train_examples.append({
            "input": f"List all my projects.",
            "output": f"Your projects are: {', '.join([p['name'] for p in data.get('projects', [])])}."
        })

        train_examples.append({
            "input": f"What is the description of '{project['name']}'?",
            "output": f"'{project['name']}' is: {project['description']}"
        })

    return train_examples

def create_dataset(examples: list) -> Dataset:
    """ Convert training examples into Hugging Face Dataset. """
    return Dataset.from_list(examples)

def tokenize_function(example: dict, tokenizer, max_length: int = 512) -> dict:
    """ Tokenize input-output pairs for training. """
    text = example["input"] + "\n\n" + example["output"]
    tokenized = tokenizer(
        text,
        padding="max_length",  # Ensures all sequences have the same length
        truncation=True,  # Truncate if longer than max_length
        max_length=max_length,
        return_tensors="pt"
    )

    # Ensure labels are properly assigned
    tokenized = {key: value.squeeze(0) for key, value in tokenized.items()}  # Remove batch dimension
    tokenized["labels"] = tokenized["input_ids"].clone()  # Use `.clone()` instead of `.copy()`

    return tokenized


def prepare_tokenized_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """ Tokenize dataset and handle empty cases. """

    if len(dataset) == 0:
        raise ValueError("Dataset is empty, cannot tokenize!")

    # Debugging: Print a sample before tokenizing
    print("Sample before tokenization:", dataset[0])

    # Apply tokenization
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=False
    )

    print("Sample after tokenization:", tokenized_dataset[0])  # Debugging

    # Ensure necessary columns are retained
    tokenized_dataset = tokenized_dataset.remove_columns(["input", "output"])
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized_dataset


def main():
    DATA_PATH = "train_data.json"
    MODEL_NAME = "deepseek-ai/deepseek-llm-7b-chat"
    LORA_PATH = "/Users/anatoly.shipitz/Projects/vLLM/lora_adapter"
    SAVE_PATH = "/Users/anatoly.shipitz/Projects/vLLM/merged_model"


    # ✅ Load and process training data
    train_examples = load_training_data(DATA_PATH)
    dataset = create_dataset(train_examples)

    # ✅ Load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Fix padding issue

    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
#         device_map={"": 0},
        device_map="auto",
        trust_remote_code=True
#         torch_dtype=torch.float16,  # Ensure GPU acceleration
#         load_in_8bit=False,
    )

    # ✅ Configure LoRA (Apply to correct layers)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # Ensure LoRA modifies attention
        lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # ✅ Confirm LoRA is trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable LoRA params: {trainable_params}")

    # ✅ Prepare dataset
    tokenized_dataset = prepare_tokenized_dataset(dataset, tokenizer)

    # ✅ Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Ensures uniform padding for better tensor compatibility
    )


    # ✅ Training settings
    training_args = TrainingArguments(
        output_dir=LORA_PATH,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        learning_rate=5e-5,
        logging_steps=1,
        save_steps=50,
        save_total_limit=1,
        report_to="none",
    )

    # ✅ Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


    print("Trainable parameters before training:", model.print_trainable_parameters())

    trainer.train()

    print("Trainable parameters after training:", model.print_trainable_parameters())

    # ✅ Save LoRA adapter
    model.save_pretrained(LORA_PATH)
    tokenizer.save_pretrained(LORA_PATH)

    # ✅ Merge LoRA for easier deployment
#     merged_model = model.merge_and_unload()
#     merged_model.save_pretrained(SAVE_PATH)
#     tokenizer.save_pretrained(SAVE_PATH)

    print("✅ LoRA training completed and models saved!")

if __name__ == "__main__":
    main()
