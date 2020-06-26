"""
Author: SaMaN
Train and test Language Model
"""
import pickle
import tensorflow as tf
from pre_train import train
from query import QueryEmbedding
from data_utils import build_word_dict, build_word_dataset

MAX_DOCUMENT_LEN = 50
FILE_DIR = "data/wikismall-simple.txt"
INPUT_FILE = open(FILE_DIR, encoding='utf-8', errors='ignore').read().split('\n')
# Training all data and Save it
print("\nBuilding dictionary..")
WORD_DICT = build_word_dict(INPUT_FILE)
print("Preprocessing dataset..")
TRAIN_X, TRAIN_Y = build_word_dataset(INPUT_FILE, WORD_DICT, MAX_DOCUMENT_LEN)
train(TRAIN_X, TRAIN_Y, WORD_DICT)
print("The LM models weights was Saved!")
print(" Language Model's Predictions: ")
tf.reset_default_graph()
QUERY = "that that that that that saman"
with open("lm_word_dict.pickle", "rb") as wd:
    WORD_DICT = pickle.load(wd)
QUERY_EMBEDDING = QueryEmbedding(QUERY, WORD_DICT)
r_F = QUERY_EMBEDDING.perplexity()
print()
"""
TO DO:
Check Max-Length, Pre-Processing(Simple va Query) and min Count .
Documentation for class's and def's.
"""
