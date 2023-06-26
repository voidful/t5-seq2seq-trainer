import math
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from module.data_processing import get_train_valid_dataset
from module.eval_metric import compute_metrics_fn

# Load model and tokenizer and Set training parameters
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# tokenizer = AutoTokenizer.from_pretrained("./training_output/Hubert/checkpoint-65699")
# model = AutoModelForSeq2SeqLM.from_pretrained("./training_output/Hubert/checkpoint-65699")

training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=20,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    predict_with_generate=True,
    learning_rate=5e-5,
    bf16=True,
    save_total_limit=10,
    learning_rate=5e-4,
    gradient_accumulation_steps=4,
)
# Define a data collator to handle tokenization
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# Load dataset
train_dataset, valid_dataset = get_train_valid_dataset(training_args, tokenizer, model.config)


def compute_metrics_middle_fn(eval_pred):
    predictions, labels = eval_pred
    labels = [i[i != -100] for i in labels]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return compute_metrics_fn(decoded_preds, decoded_labels)

def preprocess_logits_for_metrics(logits, labels):
    return logits.argmax(dim=-1)


# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_middle_fn,
    # preprocess_logits_for_metrics=preprocess_logits_for_metrics
)
# Start training
trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
