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
tokenizer = AutoTokenizer.from_pretrained("voidful/long-t5-encodec-tglobal-base")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/long-t5-encodec-tglobal-base")

training_args = Seq2SeqTrainingArguments(
    output_dir="./training_output",
    num_train_epochs=3,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=10,
    predict_with_generate=True,
    learning_rate=5e-5,
    fp16=False,
    gradient_accumulation_steps=2,
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


# Create the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_middle_fn,
)

# Start training
trainer.train()
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
