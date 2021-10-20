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

        for doc in tqdm(self.padded_encoded_docs):
            final_embeddings.append(
                np.array(np.add(doc, self.document_positional_embeddedings)))
        return final_embeddings


class LayerNormalisation():

    def __init__(self):
        pass


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=64, input_dim=(100, 512), name=None):
        super(Linear, self).__init__(name=name)
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


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
        self.linear_queries = Linear(name="linear_queries")
        self.linear_keys = Linear(name="linear_keys")
        self.linear_values = Linear(name="linear_values")

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
    print(output, end="\n\n", file=open(
        os.path.join(DIR_PATH, "log/output.txt"), "a+"))


def main():
    # Loading the Dataset
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    EMBEDDINGS_DIMENSION = 512
    MAX_SENTENCE_LENGTH = 100

    data = pd.read_csv(os.path.join(
        DIR_PATH, "data/rus.txt"), sep="\t", header=None)
    data_subset = data.iloc[:1000, 0:2]
    corpus = data_subset[0].to_list()

    print("Cleaning Corpus...")
    cleaned_corpus = clean_text(corpus)

    print("Fetching Document Embeddings...")
    encoded_docs = get_document_embeddings(
        cleaned_corpus, EMBEDDINGS_DIMENSION)

    print("Padding Document Embeddings...")
    # padded_encoded_docs = pad_encoded_docs(encoded_docs, MAX_SENTENCE_LENGTH)
    padded_encoded_docs = pad_encoded_docs(
        encoded_docs, EMBEDDINGS_DIMENSION, MAX_SENTENCE_LENGTH)

    for i in padded_encoded_docs:
        shapes = []
        for j in i:
            shapes.append(len(i))
        debug(shapes)

    # TODO: remove harcode

    input_embeddings = InputEmbedding(
        padded_encoded_docs, MAX_SENTENCE_LENGTH, EMBEDDINGS_DIMENSION)
    encoder_input = input_embeddings.call()

    debug(len(encoder_input))
    debug(len(encoder_input[0]))
    debug(len(encoder_input[0][0]))


if __name__ == "__main__":
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    # print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # gpus = tf.config.experimental.list_physical_devices("GPU")
    # tf.config.experimental.set_memory_growth(gpus[0], True)
    main()
