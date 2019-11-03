# Required packages
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import pickle
import os

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import load_model

from sentiment_train_lstm import Sentiment_train


class Sentiment(object):
    def __init__(self):
        CURENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
        DATA_PATH = os.path.join(CURENT_DIR_PATH, "data")
        model_PATH = os.path.join(CURENT_DIR_PATH, "models")
        obj_train = Sentiment_train()

        # initialise or give the model path
        self.sentiment_model = os.path.join(model_PATH, "sentiment_lstm.h5")
        try:
            # --------------to use------------------#
            self.model = load_model(self.sentiment_model)
        except:
            # ---------------to train------------------#
            self.model = obj_train.process()

    def persist(self):
        pass

