import os
import numpy as np
import pandas as pd
#import tensorflow as tf
#from tensorflow.keras.layers import Dense,Dropout
from sklearn.feature_extraction.text import CountVectorizer

class InputEmbedding():
    
    def __init__(self,corpus):
        self.input_corpus = corpus

    def run(self):
        tokenized_sentences = [sentence.split() for sentence in self.input_corpus]
        return 

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
    print(corpus)
    

if __name__ == "__main__":
    main()
