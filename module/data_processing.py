def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("voidful/NMSQA-CODE")
    train_dataset = dataset['train']
    valid_dataset = dataset['dev']

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts

        q, c = batch['hubert_100_question_unit'], batch['hubert_100_context_unit']
        a = batch['answers']
        a = convert_text_ans(a)
        for i in range(len(q)):
            if q[i] == "" and c[i] == "":
                a[i] = ""
        v_tok_q, v_tok_c = convert_vtok(q), convert_vtok(c)
        inputs = tokenizer(v_tok_q, v_tok_c, padding=True, truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Tokenize answers and create labels

        labels = tokenizer(a, padding=True, truncation=True, return_tensors="pt").input_ids
        labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
        assert len(input_ids) == len(labels)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_train_batch_size,
        cache_file_name="hubert_train_OD",
        # load_from_cache_file=True
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size,
        cache_file_name="hubert_valid_OD",
        # load_from_cache_file=True
    )
    train_dataset = train_dataset.map(
        process_data_to_model_inputs
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs
    )

    columns = ["input_ids", "labels", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    print("train_dataset", train_dataset[0])
    print("valid_dataset", valid_dataset[0])

    return train_dataset, valid_dataset

import json
# Check the mismatch between (question, context) and (answer). 
def convert_vtok(unit_code):
    for i in range(len(unit_code)):
        try:
            code = json.loads(unit_code[i])[0]['merged_code']
        except:
            continue
        v_tok = [f"v_tok_{unit}" for unit in code]
        unit_code[i] = ' '.join(v_tok) # blank is not needed
    return unit_code

def convert_text_ans(ans):
    for i in range(len(ans)):
        ans[i] = ans[i]['text'][0]
    return ans