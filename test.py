# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("voidful/long-t5-encodec-tglobal-base")

# input = "v_tok_71 v_tok_36 v_tok_49 v_tok_7 v_tok_87"
# print(tokenizer(input, truncation=True, padding=True, return_tensors="pt"))


from nlgeval import NLGEval
def _strip(s):
    return s.strip()
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR", "CIDEr"])
result_dict = dict()
labels = 'v_tok_64 v_tok_16 v_tok_50 v_tok_87 v_tok_94'
predictions = 'v_tok_64 v_tok_16 v_tok_50 v_tok_87 v_tok_94'
ref_list = list(map(list, zip(*labels)))
ref_list = [list(map(_strip, refs)) for refs in zip(*ref_list)]
refs = {idx: strippedlines for (idx, strippedlines) in enumerate(ref_list)}
hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(predictions)}
print(refs)
print(hyps)
raise

result_dict.update(nlgeval.compute_metrics(ref_list=list(map(list, zip(*labels))), hyp_list=predictions))
print(result_dict)