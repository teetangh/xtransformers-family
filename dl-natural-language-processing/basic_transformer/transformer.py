import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.python.keras.layers.advanced_activations import Softmax
from tensorflow.python.keras.layers.embeddings import Embedding
from tqdm import tqdm, trange

from abbreviations import ABBREVIATIONS
from preprocess_corpus import (clean_text, get_document_embeddings,
                               pad_encoded_docs)


class IOEmbedding(tf.keras.layers.Layer):

    def __init__(self,
                 MAX_SENTENCE_LENGTH=100,
                 EMBEDDINGS_DIMENSION=512,
                 pretrained_embeddings=None,
                 name=None):
        super(IOEmbedding, self).__init__(name=name)

        self.ABBREVIATIONS = ABBREVIATIONS
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.EMBEDDINGS_DIMENSION = EMBEDDINGS_DIMENSION
        self.document_positional_embeddedings = self.get_positional_embeddings()
        self.pretrained_embeddings = pretrained_embeddings

        if pretrained_embeddings is None:
            self.embedding_layer = Embedding(input_dim=100,
                                             output_dim=EMBEDDINGS_DIMENSION, input_length=MAX_SENTENCE_LENGTH)
        elif pretrained_embeddings is "Word2Vec" or pretrained_embeddings is "Glove":
            pass

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

    def call(self, padded_encoded_docs):
        print("Adding Positional Embeddings...")

        final_embeddings = [
            np.array(np.add(doc, self.document_positional_embeddedings))
            for doc in tqdm(padded_encoded_docs)]

        return final_embeddings


#####################################################


class Linear(tf.keras.layers.Layer):
    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8, name=None):
        super(Linear, self).__init__(name=name)
        self.self_attention = self.add_weight(
            shape=(EMBEDDINGS_DIMENSION, EMBEDDINGS_DIMENSION), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.self_attention)


# class MyDenseLayer(tf.keras.layers.Layer):
#     def __init__(self, EMBEDDINGS_DIMENSION, name=None):
#         super(MyDenseLayer, self).__init__(name=name)
#         self.EMBEDDINGS_DIMENSION = EMBEDDINGS_DIMENSION

#     def build(self):
#         self.kernel = self.add_weight(name="self-attention-weights",
#                                       shape=(self.EMBEDDINGS_DIMENSION,
#                                              self.EMBEDDINGS_DIMENSION))

#     def call(self, inputs):
#         return tf.matmul(inputs, self.kernel)

#####################################################


class ScaledDotProductAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, N_HEADS=8, name=None):
        super(ScaledDotProductAttentionLayer, self).__init__(name=name)
        self.scale_factor = N_HEADS  # TODO: Remove HARDCODE
        self.softmax = Softmax(axis=-1)

    def call(self, queries_matrix, keys_matrix, values_matrix):
        queries_keysT_product = tf.matmul(
            a=queries_matrix, b=keys_matrix, transpose_b=True)

        queries_keysT_product_scaled = queries_keysT_product / self.scale_factor

        queries_keysT_product_scaled_softmaxed = self.softmax(
            queries_keysT_product_scaled)

        return tf.matmul(
            queries_keysT_product_scaled_softmaxed, values_matrix)


class MultiHeadSelfAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8, MAX_SENTENCE_LENGTH=100, name=None):
        super(MultiHeadSelfAttentionLayer, self).__init__(name=name)
        self.EMBEDDINGS_DIMENSION = EMBEDDINGS_DIMENSION
        self.N_HEADS = N_HEADS
        self.MAX_SENTENCE_LENGTH = MAX_SENTENCE_LENGTH
        self.linear_queries = Linear(
            name="linear_queries")

        self.linear_keys = Linear(
            name="linear_keys")

        self.linear_values = Linear(
            name="linear_values")

        self.scaled_dot_product_attention = [
            ScaledDotProductAttentionLayer() for i in range(N_HEADS)]

    def call(self, encoder_input, mask=False):  # TODO: Add Encoder mask
        queries_matrix = np.array([self.linear_queries(document_embeddings)
                                   for document_embeddings in encoder_input])

        keys_matrix = np.array([self.linear_keys(document_embeddings)
                                for document_embeddings in encoder_input])

        values_matrix = np.array([self.linear_values(document_embeddings)
                                  for document_embeddings in encoder_input])

        queries_matrix_head = np.split(queries_matrix, self.N_HEADS, axis=-1)
        keys_matrix_head = np.split(keys_matrix, self.N_HEADS, axis=-1)
        values_matrix_head = np.split(values_matrix, self.N_HEADS, axis=-1)

        print(np.array(queries_matrix_head).shape)

        print("Concatenating Heads")
        concat_attention = np.concatenate(
            [self.scaled_dot_product_attention[i](
                queries_matrix_head[i], keys_matrix_head[i], values_matrix_head[i])
             for i in trange(self.N_HEADS)], axis=-1)

        return np.array(concat_attention)
        # .reshape(-1, self.MAX_SENTENCE_LENGTH, self.EMBEDDINGS_DIMENSION)


class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8,  name=None):
        super(EncoderBlock, self).__init__(name=name)
        self.multihead_self_attention = MultiHeadSelfAttentionLayer()
        self.layer_normalisation = LayerNormalization(
            axis=-1)
        self.feed_forward = Linear(name="feed_forward")

    def call(self, encoder_input):

        self_attention_output = self.multihead_self_attention(
            encoder_input)

        layer_normalisation_output = self.layer_normalisation(
            self_attention_output)

        encoder_intermediate_output = encoder_input + layer_normalisation_output

        feed_forward_output = [self.feed_forward(document_embeddings)
                               for document_embeddings in encoder_intermediate_output]

        encoder_output = encoder_input + feed_forward_output

        return encoder_output


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8, name=None):
        super(DecoderBlock, self).__init__(name=name)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8, name=None):
        super(Encoder, self).__init__(name=name)
        self.encoder_block = EncoderBlock(
            EMBEDDINGS_DIMENSION=512, N_HEADS=8, name="encoder")

    def call(self, encoder_input):
        return self.encoder_block(encoder_input)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, EMBEDDINGS_DIMENSION=512, N_HEADS=8, name=None):
        super(Decoder, self).__init__(name=name)


class Transformer(tf.keras.Model):

    def __init__(self,
                 MAX_SENTENCE_LENGTH=100,
                 EMBEDDINGS_DIMENSION=512,
                 N_HEADS=8, name=None):
        super(Transformer, self).__init__(name=name)

        self.input_embeddings = IOEmbedding(
            MAX_SENTENCE_LENGTH, EMBEDDINGS_DIMENSION, pretrained_embeddings="Word2Vec")
        self.encoder = Encoder(EMBEDDINGS_DIMENSION=512, N_HEADS=8)
        # self.decoder = Decoder()

    def call(self, padded_encoded_docs):
        encoder_input = self.input_embeddings(padded_encoded_docs)
        temp = self.encoder(encoder_input)
        # assert(np.array(temp).shape == (400, 100, 512))
        return temp


def debug(output):
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    print(output, end="\n\n", file=open(
        os.path.join(DIR_PATH, "log/output.txt"), "a+"))


def main():
    # Loading the Dataset
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    EMBEDDINGS_DIMENSION = 512
    MAX_SENTENCE_LENGTH = 100
    N_HEADS = 8
    # QUERY_SIZE = EMBEDDINGS_DIMENSION / N_HEADS

    data = pd.read_csv(os.path.join(
        DIR_PATH, "data/rus.txt"), sep="\t", header=None)
    data_subset = data.iloc[:200, 0:2]
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

    transformer = Transformer(
        EMBEDDINGS_DIMENSION=EMBEDDINGS_DIMENSION,
        MAX_SENTENCE_LENGTH=MAX_SENTENCE_LENGTH,
        N_HEADS=N_HEADS, name="transformer")
    transformer(padded_encoded_docs)


if __name__ == "__main__":
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    # tf.config.gpu.set_per_process_memory_fraction(0.4)
#
    main()
