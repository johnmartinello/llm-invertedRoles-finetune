import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json

def load_human_user_dataset(split):
    """Load the human-user fine-tuning dataset"""
    data = []
    with open(f"data/finetuning_dataset_{split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

def format_for_training(example):
    """Format for causal language modeling (predict next token)"""
    # Combine chat_history + target_response as one sequence
    # Chat history contains AI assistant messages, target is human user response
    full_text = example["chat_history"] + "\n" + example["target_response"]
    return {"text": full_text}

def setup_model_and_tokenizer():
    """Setup model with QLoRA"""
    model_name = "Qwen/Qwen2.5-7B"
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 4-bit NF4 quantization with double quant and bfloat16 compute if available
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    model = prepare_model_for_kbit_training(model)
    # Reduce VRAM and avoid checkpointing warning
    if hasattr(model, "config"):
        model.config.use_cache = False
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

def main():
    """
    Train a model to act as a human user responding to AI assistant questions.
    
    Training format:
    - Chat History: AI assistant messages (context)
    - Target Response: Human user messages (what the model learns to generate)
    
    Example:
    - Input: "Assistant: How can I help you today?"
    - Target: "User: I want the recipe for a chocolate cake"
    """
    # Load datasets
    train_dataset = load_human_user_dataset("train")
    val_dataset = load_human_user_dataset("validation")
    
    # Format for training
    train_dataset = train_dataset.map(format_for_training)
    val_dataset = val_dataset.map(format_for_training)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize with target-only loss (mask history)
    def tokenize_and_mask(example, max_len=512):
        # Chat history contains AI assistant messages (context)
        # Target response is the human user message (what we want the model to learn)
        prompt = example["chat_history"].strip() if example.get("chat_history") else ""
        target = example["target_response"].strip() if example.get("target_response") else ""
        sep = "\n" if prompt else ""
        full = (prompt + sep + target)

        toks = tokenizer(full, truncation=True, max_length=max_len)

        # Compute prompt length in tokens for masking
        # Only compute loss on the target (human user response), not the context
        prompt_ids = tokenizer(prompt + ("\n" if prompt else ""), add_special_tokens=False)["input_ids"] if prompt else []
        prompt_len = len(prompt_ids)

        labels = toks["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100  # Mask the context (AI assistant messages)

        toks["labels"] = labels
        return toks

    max_len = 512
    train_dataset = train_dataset.map(lambda e: tokenize_and_mask(e, max_len))
    val_dataset = val_dataset.map(lambda e: tokenize_and_mask(e, max_len))

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./qlora-adapters",
        # Training length
        max_steps=2000,  # cap total steps (~8h at 15s/step)
        num_train_epochs=1,  # just in case, but max_steps dominates
        # Batch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,  # effective batch = 8
        # Learning rate
        learning_rate=3e-4,  # slightly higher for LoRA
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        # Logging & evaluation
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Precision
        fp16=True,  # 3060 Ti works best with fp16
        bf16=False, # disable bf16 (your GPU doesn’t support it)
        # Memory
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_eval_batch_size=1,
        group_by_length=True,
        dataloader_num_workers=2,
        # Best model metric
        metric_for_best_model="eval_perplexity",
        greater_is_better=False,
    )

    
    # Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    class PerplexityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None and "eval_loss" in metrics and metrics["eval_loss"] is not None:
                try:
                    metrics["eval_perplexity"] = float(torch.exp(torch.tensor(metrics["eval_loss"])) .item())
                except Exception:
                    metrics["eval_perplexity"] = float("inf")
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[PerplexityCallback()],
    )
    
    # Train
    trainer.train()
    
    # Save adapters
    trainer.save_model()

if __name__ == "__main__":
    main()