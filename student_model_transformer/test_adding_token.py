import torch
import os
import wget
import tarfile
import shutil
import codecs
import youtokentome
import math
import random
from tqdm import tqdm

data_folder = '/Users/yuumu/Desktop/Master_Sem1/Master_project/Translation Model/data_special'  


def download_data(data_folder):
    """
    Downloads the training, validation, and test files for WMT '14 en-de translation task.

    Training: Europarl v7, Common Crawl, News Commentary v9
    Validation: newstest2013
    Testing: newstest2014

    The homepage for the WMT '14 translation task, https://www.statmt.org/wmt14/translation-task.html, contains links to
    the datasets.

    :param data_folder: the folder where the files will be downloaded

    """
    train_urls = ["http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
                  "https://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
                  "http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz"]
    

    print("\n\nThis may take a while.")

    # Create a folder to store downloaded TAR files
    if not os.path.isdir(os.path.join(data_folder, "tar files")):
        os.mkdir(os.path.join(data_folder, "tar files"))
    # Create a fresh folder to extract downloaded TAR files; previous extractions deleted to prevent tarfile module errors
    if os.path.isdir(os.path.join(data_folder, "extracted files")):
        shutil.rmtree(os.path.join(data_folder, "extracted files"))
        os.mkdir(os.path.join(data_folder, "extracted files"))

    # Download and extract training data
    for url in train_urls:
        filename = url.split("/")[-1]
        if not os.path.exists(os.path.join(data_folder, "tar files", filename)):
            print("\nDownloading %s..." % filename)
            wget.download(url, os.path.join(data_folder, "tar files", filename))
        print("\nExtracting %s..." % filename)
        tar = tarfile.open(os.path.join(data_folder, "tar files", filename))
        members = [m for m in tar.getmembers() if "de-en" in m.path]
        tar.extractall(os.path.join(data_folder, "extracted files"), members=members)

    # Download validation and testing data using sacreBLEU since we will be using this library to calculate BLEU scores
    print("\n")
    os.system("sacrebleu -t wmt13 -l en-de --echo src > '" + os.path.join(data_folder, "val.en") + "'")
    os.system("sacrebleu -t wmt13 -l en-de --echo ref > '" + os.path.join(data_folder, "val.de") + "'")
    print("\n")
    os.system("sacrebleu -t wmt14/full -l en-de --echo src > '" + os.path.join(data_folder, "test.en") + "'")
    os.system("sacrebleu -t wmt14/full -l en-de --echo ref > '" + os.path.join(data_folder, "test.de") + "'")

    # Move files if they were extracted into a subdirectory
    for dir in [d for d in os.listdir(os.path.join(data_folder, "extracted files")) if
                os.path.isdir(os.path.join(data_folder, "extracted files", d))]:
        for f in os.listdir(os.path.join(data_folder, "extracted files", dir)):
            shutil.move(os.path.join(data_folder, "extracted files", dir, f),
                        os.path.join(data_folder, "extracted files"))
        os.rmdir(os.path.join(data_folder, "extracted files", dir))



def prepare_data(data_folder, euro_parl=True, common_crawl=True, news_commentary=True, min_length=3, max_length=100,
                 max_length_ratio=1.5, retain_case=True):
    """
    Filters and prepares the training data, trains a Byte-Pair Encoding (BPE) model.

    :param data_folder: the folder where the files were downloaded
    :param euro_parl: include the Europarl v7 dataset in the training data?
    :param common_crawl: include the Common Crawl dataset in the training data?
    :param news_commentary: include theNews Commentary v9 dataset in the training data?
    :param min_length: exclude sequence pairs where one or both are shorter than this minimum BPE length
    :param max_length: exclude sequence pairs where one or both are longer than this maximum BPE length
    :param max_length_ratio: exclude sequence pairs where one is much longer than the other
    :param retain_case: retain case?
    """
    # Read raw files and combine
    german = list()
    english = list()
    files = list()
    assert euro_parl or common_crawl or news_commentary, "Set at least one dataset to True!"
    if euro_parl:
        files.append("europarl-v7.de-en")
    if common_crawl:
        files.append("commoncrawl.de-en")
    if news_commentary:
        files.append("news-commentary-v9.de-en")
    print("\nReading extracted files and combining...")
    for file in files:
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".de"), "r", encoding="utf-8") as f:
            if retain_case:
                german.extend(f.read().split("\n"))
            else:
                german.extend(f.read().lower().split("\n"))
        with codecs.open(os.path.join(data_folder, "extracted files", file + ".en"), "r", encoding="utf-8") as f:
            if retain_case:
                english.extend(f.read().split("\n"))
            else:
                english.extend(f.read().lower().split("\n"))
        assert len(english) == len(german)

    # Write to file so stuff can be freed from memory
    print("\nWriting to single files...")
    bleu = ['BLEU10','BLEU20','BLEU30','BLEU40','BLEU50','BLEU60','BLEU70','BLEU80','BLEU90','BLEU100']
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        for english_sentence in english:
            ran = random.choice(bleu)
            f.write(english_sentence + " " + ran +"\n")
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        for german_sentence in german:
            ran = random.choice(bleu)
            f.write(german_sentence + " " + ran + "\n")
    with codecs.open(os.path.join(data_folder, "train.ende"), "w", encoding="utf-8") as f:
        for english_sentence in english:
            ran = random.choice(bleu)
            f.write(english_sentence + " " + ran +"\n")
        for german_sentence in german:
            ran = random.choice(bleu)
            f.write(german_sentence + " " + ran + "\n")
    del english, german  # free some RAM

    # Perform BPE
    print("\nLearning BPE...")
    youtokentome.BPE.train(data=os.path.join(data_folder, "train.ende"), vocab_size=37000,
                           model=os.path.join(data_folder, "bpe_withs.model"))

     # Load BPE model
    print("\nLoading BPE model...")
    bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe_withs.model"))

    # Re-read English, German
    print("\nRe-reading single files...")
    with codecs.open(os.path.join(data_folder, "train.en"), "r", encoding="utf-8") as f:
        english = f.read().split("\n")
    with codecs.open(os.path.join(data_folder, "train.de"), "r", encoding="utf-8") as f:
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
    print("\nRe-writing filtered sentences to single files...")
    os.remove(os.path.join(data_folder, "train.en"))
    os.remove(os.path.join(data_folder, "train.de"))
    os.remove(os.path.join(data_folder, "train.ende"))
    with codecs.open(os.path.join(data_folder, "train.en"), "w", encoding="utf-8") as f:
        f.write("\n".join(english))
    with codecs.open(os.path.join(data_folder, "train.de"), "w", encoding="utf-8") as f:
        f.write("\n".join(german))
    del english, german, bpe_model, pairs

    print("\n...DONE!\n")




download_data(data_folder=data_folder)

prepare_data(data_folder=data_folder,
             euro_parl=True,
             common_crawl=True,
             news_commentary=True,
             min_length=3,
             max_length=150,
             max_length_ratio=2.,
             retain_case=True)


# bpe_model = youtokentome.BPE(model=os.path.join(data_folder, "bpe.model"))
# ans = bpe_model.encode("Hello how are you? BLEU70",output_type=youtokentome.OutputType.SUBWORD)
# print(ans)