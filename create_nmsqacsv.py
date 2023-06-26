import datasets
import os
import IPython.display as ipd
from tqdm import tqdm

train_set = datasets.load_dataset("voidful/NMSQA_audio", split='train')
dev_set = datasets.load_dataset("voidful/NMSQA_audio", split='dev')
train_dir, dev_dir = "./NMSQA-train-wav", "./NMSQA-dev-wav"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
with open("NMSQA-train.csv", "w") as f:
    f.writelines("path,text\n")
with open("NMSQA-dev.csv", "w") as f:
    f.writelines("path,text\n")

for data in tqdm(train_set):
    try:
        audio_data = data['content_segment_audio_path']['array']
        audio = ipd.Audio(data=audio_data, autoplay=False, rate=24000)
    except:
        continue
    gt_text = data['content_segment_normalized_text']
    ids = data['id']
    file_name = ids + '.wav'
    with open(os.path.join(train_dir, file_name), "wb") as f:
        f.write(audio.data)
    with open("NMSQA-train.csv", "a") as f:
        f.writelines(os.path.join(train_dir, file_name) + ',"' + gt_text + '"\n')
