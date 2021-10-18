import os
import numpy as np
import pandas as pd

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#import tensorflow as tf
#from tensorflow.keras.layers import Dense,Dropout
from sklearn.feature_extraction.text import CountVectorizer

class InputEmbedding():
    
    def __init__(self,corpus):
        self.input_corpus = corpus

    def clean_text(self,df):
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

    def get_postition_encoding(self,all_text):
        pass

    def run(self):
        self.cleaned_text = self.clean_text(self.input_corpus)
        self.tokenized_sentences = [sentence.split() for sentence in self.cleaned_text]
        self.position_encoded_text = get_position_encoding(self.tokenized_sentences)

class ScaledDotProdcutAttention():
    def __init__(self):
        pass

class MultiHeadAttention():
    def __init__(self):
        pass

class MaskedMultiHeadAttention():
    def __init__(self):
        pass

class Encoder():
    def __init__(self):
        pass

class Decoder():
    def __init__(self):
        pass

class Transformer():
    def __init__(self):
        pass

def main():
    #EMBEDDING_DIR = ...
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(DIR_PATH,"data/rus.txt"),sep="\t",header=None)
    data = data.iloc[:,0:2]
    corpus = data[0].to_list()
    
    input_embeddings = InputEmbedding(corpus)
    input_embeddings.run()
    print(input_embeddings.tokenized_sentences)

if __name__ == "__main__":
    main()
