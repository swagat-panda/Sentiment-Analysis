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

from numpy import array
from numpy import asarray
from numpy import zeros

TAG_RE = re.compile(r'<[^>]+>')


class Sentiment_train:
    def __init__(self):
        CURENT_DIR_PATH = os.path.abspath(os.path.dirname(__file__))
        DATA_PATH = os.path.join(CURENT_DIR_PATH, "data")
        model_PATH = os.path.join(CURENT_DIR_PATH, "models")
        glove_model_path = os.path.join(model_PATH, "glove_model")

        ###
        self.data_file = os.path.join(DATA_PATH, "IMDB Dataset.csv")
        self.movie_reviews = pd.read_csv(self.data_file)
        self.movie_reviews.isnull().values.any()
        self.glove_file = os.path.join(glove_model_path, "glove.6B.100d.txt")
        self.glove_file = open(self.glove_file, encoding="utf8")

    def remove_tags(self, text):
        return TAG_RE.sub('', text)

    def preprocess_text(self, sen):
        # Removing html tags
        sentence = self.remove_tags(sen)

        # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)

        # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

        # Removing multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence

    def lstm_model(self, vocab_size, embedding_matrix, maxlen):
        model = Sequential()
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)
        model.add(LSTM(128))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        return model

    def train(self, model, X_train, y_train, X_test, y_test):
        history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_split=0.2)
        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])
        return model

    def process(self):
        X = []
        sentences = list(self.movie_reviews['review'])
        for sen in sentences:
            X.append(self.preprocess_text(sen))

        y = self.movie_reviews['sentiment']

        y = np.array(list(map(lambda x: 1 if x == "positive" else 0, y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(X_train)

        X_train = tokenizer.texts_to_sequences(X_train)
        X_test = tokenizer.texts_to_sequences(X_test)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        maxlen = 100

        X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

        embeddings_dictionary = dict()
        for line in self.glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
        self.glove_file.close()

        embedding_matrix = zeros((vocab_size, 100))
        for word, index in tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        model = self.lstm_model(vocab_size, embedding_matrix, maxlen)
        print(model.summary())
        model = self.train(model, X_train, y_train, X_test, y_test)
        return model
