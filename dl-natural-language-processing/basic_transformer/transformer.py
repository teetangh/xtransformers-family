import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm, trange

from abbreviations import ABBREVIATIONS
from preprocess_corpus import (clean_text, get_document_embeddings,
                               pad_encoded_docs)


class InputEmbedding():

    def __init__(self, padded_encoded_docs, MAX_SENTENCE_LENGTH=100, EMBEDDINGS_DIMENSION=512):
        self.padded_encoded_docs = padded_encoded_docs
        self.CORPUS_SIZE = len(padded_encoded_docs)
        self.ABBREVIATIONS = ABBREVIATIONS
        # since word2vec # TODO: make generic later
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.EMBEDDINGS_DIMENSION = EMBEDDINGS_DIMENSION
        self.document_positional_embeddedings = self.get_positional_embeddings()

    def get_positional_embeddings(self):
        print("Calculating Positional Embeddings...")
        positional_embeddings = np.zeros(
            (self.MAX_SENTENCE_LENGTH, self.EMBEDDINGS_DIMENSION))

        for position in trange(self.MAX_SENTENCE_LENGTH):
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
        print("Adding Positional Embeddings...")

        final_embeddings = []

        print(len(self.padded_encoded_docs))
        print(len(self.padded_encoded_docs[0]))
        print(len(self.padded_encoded_docs[0][0]))

        # print(len(document_positional_embeddedings))
        # print(len(document_positional_embeddedings[0]))
        # print(len(document_positional_embeddedings[0][0]))

        for doc in tqdm(self.padded_encoded_docs):
            np.add(np.asmatrix(doc), np.mat(
                self.document_positional_embeddedings))

        # for doc in tqdm(self.padded_encoded_docs):
        #     final_embedding = []
        #     for embedding in tqdm(self.document_positional_embeddedings):
        #         print("len(doc) ", len(doc))
        #         print("len(embedding) ", len(embedding))
        #         final_embedding.append(np.sum([doc, embedding], axis=0))
        #     final_embeddings.append(final_embedding)
        # return final_embeddings


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


def debug(output):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    print(output, end="\n\n\n\n\n\n", file=open(
        os.path.join(DIR_PATH, "log/output.txt"), "a+"))


def main():
    # Loading the Dataset
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    EMBEDDINGS_DIMENSION = 512
    MAX_SENTENCE_LENGTH = 100

    data = pd.read_csv(os.path.join(
        DIR_PATH, "data/rus.txt"), sep="\t", header=None)
    data_subset = data.iloc[:10000, 0:2]
    corpus = data_subset[0].to_list()

    print("Cleaning Corpus...")
    cleaned_corpus = clean_text(corpus)

    print("Fetching Document Embeddings...")
    encoded_docs = get_document_embeddings(
        cleaned_corpus, EMBEDDINGS_DIMENSION)

    print("Padding Document Embeddings...")
    padded_encoded_docs = pad_encoded_docs(encoded_docs, MAX_SENTENCE_LENGTH)

    # for i in padded_encoded_docs:
    #     shapes = []
    #     for j in i:
    #         shapes.append(len(i))
    #     debug(shapes)

    # TODO: remove harcode

    print(len(padded_encoded_docs))
    print(len(padded_encoded_docs[0]))
    print(len(padded_encoded_docs[0][0]))

    input_embeddings = InputEmbedding(
        padded_encoded_docs, MAX_SENTENCE_LENGTH, EMBEDDINGS_DIMENSION)
    encoder_input = input_embeddings.call()


if __name__ == "__main__":
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    # print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
