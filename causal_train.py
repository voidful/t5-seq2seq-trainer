from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)
from module.data_processing import get_train_valid_dataset
from module.eval_metric import compute_metrics_fn
import json
import numpy as np
import math

# Load model and tokenizer and Set training parameters
model = AutoModelForCausalLM.from_pretrained("voidful/stablelm-tuned-alpha-3b-unit")
tokenizer = AutoTokenizer.from_pretrained("voidful/stablelm-tuned-alpha-3b-unit")

training_args = TrainingArguments(
    output_dir="./training_output/stablelm-tuned-alpha-3b-unit",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=3e-3,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=5e-4,
    fp16=True,
    gradient_accumulation_steps=8,
)

# Load dataset
train_dataset, valid_dataset = get_train_valid_dataset(training_args, tokenizer, model.config)
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_fn,
    # data_collator=data_collator,
    # prediction_loss_only=True,
    # post_process_function=preprocess_logits_for_metrics
)
# Train model
trainer.train()
# Evaluate model
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
