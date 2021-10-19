import multiprocessing
import os
import re
import string

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

from abbreviations import ABBREVIATIONS


def load_nlp_libraries():
    nltk.download('punkt')
    nltk.download('stopwords')
    print("loaded stop words")
    word2vec_model = api.load("word2vec-google-news-300")
    print("loaded word2vec")


def clean_text(df):
    cleaned_text = list()
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

        text = [re.sub(abbreviation[0], abbreviation[1], str(text))
                for abbreviation in ABBREVIATIONS]

        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", str(text))

        # tokenize over split return list rather than strings
        tokens = word_tokenize(text)

        # remove all the punctuations (safety)
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]

        # Remove everything that is not an alphabet
        words = [word for word in stripped if word.isalpha()]

        # Remove the stop words (set of words that do not provide meaningful information)
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        # words = [w for w in words if not w in stop_words]

        # Use Porter Stemmer to stem the words to root words (necessary for sentiment analysis and text processing)
        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)  # join the tokens (make a string)
        cleaned_text.append(words)  # add it to the large list

    return cleaned_text


def get_document_embeddings(cleaned_corpus, EMBEDDING_SIZE=10):
    # Loading Word2Vec
    # TODO: Use 
    # EMBEDDING_DIR = ...
    NUM_CORES = multiprocessing.cpu_count()

    w2v_model = None
    model_path = f"models/word2vec.model"

    # Load the Word2Vec model if it exists
    if os.path.exists(model_path):
        w2v_model = Word2Vec.load(model_path)
    else:
        w2v_model = Word2Vec(
            sentences=cleaned_corpus,
            vector_size=EMBEDDING_SIZE,
            min_count=1,
            window=5,
            workers=NUM_CORES,
            seed=1337
        )

    # w2v_model.wv.most_similar(positive="program")
    # encoded_docs is a 3d list
    # TODO: (if some lists are empty that's a problem in embedding )
    encoded_docs = [[w2v_model.wv[word] for word in post]
                    for post in cleaned_corpus]

    return encoded_docs


def pad_encoded_docs(encoded_docs, MAX_LENGTH=10):
    padded_posts = []
    for post in encoded_docs:
        # Pad short posts with alternating min/max

        # TODO: Find a better approach
        if len(post) == 0:
            post = [0] * MAX_LENGTH

        if len(post) < MAX_LENGTH:

            # Method 1
            pointwise_min = np.minimum.reduce(post)
            pointwise_max = np.maximum.reduce(post)
            padding = [pointwise_max, pointwise_min]

            # Method 2
            pointwise_avg = np.mean(post)
            padding = [pointwise_avg]

            # print(post)
            post += padding * int(np.ceil((MAX_LENGTH - len(post) / 2.0)))

        # Shorten long posts or those odd number length posts we padded to 51
        if len(post) > MAX_LENGTH:
            post = post[:MAX_LENGTH]

        print(post)
        # Add the post to our new list of padded posts
        padded_posts.append(post)

    return padded_posts
