import multiprocessing
import os
import re
import string
from codecs import encode
from operator import pos

import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.util import pr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.ops.gen_array_ops import size

from abbreviations import ABBREVIATIONS
from preprocess_corpus import (clean_text, get_document_embeddings,
                               load_nlp_libraries, pad_encoded_docs)


class InputEmbedding(tf.keras.layers.Layer):

    def __init__(self, corpus):
        self.input_corpus = corpus
        self.CORPUS_SIZE = len(corpus)
        self.ABBREVIATIONS = ABBREVIATIONS
        self.EMBEDDINGS_DIMENSION = 300  # since word2vec # TODO: make generic later

    def tokenize_text(self,):
        pass

    def get_word_embeddings(self, tokenized_sentences):
        # sentence_vectors = self.vectorizer.fit_transform(
        #     self.input_corpus)
        # print(sentence_vectors.shape)
        # self.EMBEDDINGS_DIMENSION = sentence_vectors.shape[1]

        all_embeddings = []
        for tokenized_sentence in tokenized_sentences:
            sentence_embeddings = []
            for token in tokenized_sentence:
                sentence_embeddings.append(word2vec_model[token])
            all_embeddings.append(sentence_embeddings)

        return all_embeddings

    def get_positional_embeddings(self):

        positional_embeddings = np.zeros(
            (self.CORPUS_SIZE, self.EMBEDDINGS_DIMENSION + 1))

        for position in range(self.CORPUS_SIZE):
            for i in range(0, self.EMBEDDINGS_DIMENSION, 2):
                positional_embeddings[position, i] = (
                    np.sin(position / (10000 ** ((2*i) / self.EMBEDDINGS_DIMENSION)))
                )
                positional_embeddings[position, i + 1] = (
                    np.cos(
                        position / (10000 ** ((2 * (i + 1)) / self.EMBEDDINGS_DIMENSION)))
                )

        # print(positional_embeddings.shape)
        return tf.convert_to_tensor(positional_embeddings)

    def call(self):
        # # TODO: remove highlight
        self.tokenized_sentences = [sentence.split()
                                    for sentence in self.cleaned_text]
        #                             for sentence in self.input_corpus]

        # input_embeddings = self.get_word_embeddings(self.tokenized_sentences)
        positional_embeddedings = self.get_positional_embeddings()

        return np.add(input_embeddings, positional_embeddedings)


class LayerNormalisation():

    def __init__(self):
        pass


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, queries_vector,
                 keys_vector,
                 values_vector, name=None):
        super(ScaledDotProductAttentionLayer, self).__init__(name=name)
        self.queries_vector = queries_vector
        self.keys_vector = keys_vector
        self.values_vector = values_vector

    def call(self):
        pass


class MultiHeadSelfAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, name=None):
        super(MultiHeadSelfAttentionLayer, self).__init__(name=name)
        self.vector_dimension = 64
        self.queries_vector = Dense(units=self.vector_dimension,
                                    activation="relu",
                                    kernel_initializer="random_normal",
                                    bias_initializer="random_normal")
        self.keys_vector = Dense(units=self.vector_dimension,
                                 activation="relu",
                                 kernel_initializer="random_normal",
                                 bias_initializer="random_normal")
        self.values_vector = Dense(units=self.vector_dimension,
                                   activation="relu",
                                   kernel_initializer="random_normal",
                                   bias_initializer="random_normal")
        self.scaled_dot_product_attention = ScaledDotProductAttentionLayer(
            self.queries_vector, self.keys_vector, self.values_vector)

    def call(self):
        pass


# class MaskedMultiHeadAttention():
#     def __init__(self):
#         pass


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(EncoderBlock, self).__init__(name=name)
        self.multihead_self_attention_layer = MultiHeadSelfAttentionLayer()

    def call(self, encoder_input):

        for document_embeddings in encoder_input:
            pass


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(DecoderBlock, self).__init__(name=name)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Encoder, self).__init__(name=name)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, name=None):
        super(Decoder, self).__init__(name=name)


class Transformer(tf.keras.Model):

    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()


def main():
    # Loading the Dataset
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(
        DIR_PATH, "data/rus.txt"), sep="\t", header=None)
    data_subset = data.iloc[:10000, 0:2]

    corpus = data_subset[0].to_list()
    cleaned_corpus = clean_text(corpus)
    encoded_docs = get_document_embeddings(cleaned_corpus)
    padded_encoded_docs = pad_encoded_docs(encoded_docs)

    # TODO: remove harcode

    # input_embeddings = InputEmbedding(cleaned_corpus)
    # encoder_input = input_embeddings.call()


if __name__ == "__main__":

    main()
