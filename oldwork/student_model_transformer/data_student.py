import torch
import os
import wget
import tarfile
import shutil
import codecs
import youtokentome
import math
from tqdm import tqdm

data_folder = '/Users/yuumu/Desktop/Master_Sem1/Master_project/Translation Model/data' 
min_length=3
max_length=100
max_length_ratio=1.5


bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))

# Re-read English, German
print("\nRe-reading single files...")
with codecs.open(os.path.join(data_folder, "teacher_output.txt"), "r", encoding="utf-8") as f:
    english = f.read().split("\n")
with codecs.open(os.path.join(data_folder, "teacher_input.txt"), "r", encoding="utf-8") as f:
    german = f.read().split("\n")

# Filter
print("\nFiltering...")
pairs = list()
for en, de in tqdm(zip(english, german), total=len(english)):
    en_tok = bpe_model.encode(en, output_type=youtokentome.OutputType.ID)
    de_tok = bpe_model.encode(de, output_type=youtokentome.OutputType.ID)
    len_en_tok = len(en_tok)
    len_de_tok = len(de_tok)
    if min_length < len_en_tok < max_length and \
            min_length < len_de_tok < max_length and \
            1. / max_length_ratio <= len_de_tok / len_en_tok <= max_length_ratio:
        pairs.append((en, de))
    else:
        continue
print("\nNote: %.2f per cent of en-de pairs were filtered out based on sub-word sequence length limits." % (100. * (
        len(english) - len(pairs)) / len(english)))

# Rewrite files
english, german = zip(*pairs)
# print("\nRe-writing filtered sentences to single files...")
# os.remove(os.path.join(data_folder, "train.en"))
# os.remove(os.path.join(data_folder, "train.de"))
# os.remove(os.path.join(data_folder, "train.ende"))
with codecs.open(os.path.join(data_folder, "student_train.en"), "w", encoding="utf-8") as f:
    f.write("\n".join(english))
with codecs.open(os.path.join(data_folder, "student_train.de"), "w", encoding="utf-8") as f:
    f.write("\n".join(german))
del english, german, bpe_model, pairs

print("\n...DONE!\n")
