"""
Author: SaMaN
"""
import os
import re
import pickle
import collections
import numpy as np
from nltk.tokenize import word_tokenize


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def build_word_dict(input_file):
    if not os.path.exists("lm_word_dict.pickle"):
        words = list()
        for content in input_file:
            for word in word_tokenize(clean_str(content)):
                words.append(word)
        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, count in word_counter:
            if count > 1:
                word_dict[word] = len(word_dict)
        with open("lm_word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)
    else:
        with open("lm_word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)
    return word_dict


def build_word_dataset(input_file, word_dict, document_max_len):
    input2int = list(map(lambda d: word_tokenize(clean_str(d)), input_file))
    input2int = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), input2int))
    input2int = list(map(lambda d: d[:document_max_len], input2int))
    input2int = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], input2int))
    target_text = list(map(lambda d: d, list(input2int)))
    return input2int[:70], target_text[:70]


def batch_iter(input_file, target, batch_size, num_epochs):
    input_text = np.array(input_file)
    targets = np.array(target)
    num_batches_per_epoch = (len(input_text) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(input_text))
            yield input_text[start_index:end_index], targets[start_index:end_index]
