import os
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')


#import tensorflow as tf
#from tensorflow.keras.layers import Dense,Dropout


class InputEmbedding():

    def __init__(self, corpus):
        self.input_corpus = corpus
        self.CORPUS_SIZE = len(corpus)
        self.vectorizer = CountVectorizer()

    def clean_text(self, df):
        all_text = list()
        lines = df
        PS = PorterStemmer()  # Takes only the root words

        pattern = re.compile(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)

        for text in lines:
            text = text.lower()  # convert to lower case
            text = pattern.sub('', text)  # remove all the links
            text = emoji_pattern.sub(r"", text)

            text = re.sub(r"i'm", "i am", text)
            text = re.sub(r"he's", "he is", text)
            text = re.sub(r"she's", "she is", text)
            text = re.sub(r"that's", "that is", text)
            text = re.sub(r"what's", "what is", text)
            text = re.sub(r"where's", "where is", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"won't", "will not", text)
            text = re.sub(r"don't", "do not", text)
            text = re.sub(r"did't", "did not", text)
            text = re.sub(r"can't", "can not", text)
            text = re.sub(r"it's", "it is", text)
            text = re.sub(r"couldn't", "could not", text)
            text = re.sub(r"have't", "have not", text)

            text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)

            # tokenize over split return list rather than strings
            tokens = word_tokenize(text)

            # remove all the punctuations (safety)
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]

            # Remove everything that is not an alphabet
            words = [word for word in stripped if word.isalpha()]

            # Remove the stop words (set of words that do not provide meaningful information)
            # stop_words = set(stopwords.words("english"))
            # stop_words.discard("not")
            # words = [w for w in words if not w in stop_words]

            # Use Porter Stemmer to stem the words to root words (necessary for sentiment analysis and text processing)
            # words = [PS.stem(w) for w in words if not w in stop_words]
            words = ' '.join(words)  # join the tokens (make a string)
            all_text.append(words)  # add it to the large list

        return all_text

    def get_word_embeddings(self, tokenized_sentences):
        sentence_vectors = self.vectorizer.fit_transform(
            self.input_corpus)
        # print(sentence_vectors.shape)
        self.EMBEDDINGS_DIMENSION = sentence_vectors.shape[1]
        return sentence_vectors

    def get_positional_embeddings(self, input_embeddings):

        positional_embeddings = np.zeros(
            (self.CORPUS_SIZE + 1, self.EMBEDDINGS_DIMENSION + 1))

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
        return positional_embeddings

    def run(self):
        # self.cleaned_text = self.clean_text(self.input_corpus)

        # # TODO: remove highlight
        # self.tokenized_sentences = [sentence.split()
        #                             # for sentence in self.cleaned_text]
        #                             for sentence in self.input_corpus]

        input_embeddings = self.get_word_embeddings(self.input_corpus)
        # print("input embeddings")
        # for i in range(10):
        #     print(input_embeddings.toarray())

        positional_embedded_text = self.get_positional_embeddings(
            input_embeddings)
        # print("\n positional_embedded_text \n", positional_embedded_text)

        return positional_embedded_text


class ScaledDotProductAttention():
    def __init__(self):
        pass


class MultiHeadAttention():
    def __init__(self):
        self.scaled_dot_product_attention = ScaledDotProductAttention()


class MaskedMultiHeadAttention():
    def __init__(self):
        pass


class Encoder():

    def __init__(self, encoder_input):
        self.encoder_input = encoder_input


class Decoder():
    def __init__(self):
        pass


class Transformer():
    def __init__(self):
        pass


def main():
    #EMBEDDING_DIR = ...
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(
        DIR_PATH, "data/rus.txt"), sep="\t", header=None)
    data = data.iloc[:10000, 0:2]
    corpus = data[0].to_list()

    # TODO: remove harcode
    input_embeddings = InputEmbedding(corpus)
    encoder_input = input_embeddings.run()
    # print(encoder_input)


if __name__ == "__main__":
    main()
