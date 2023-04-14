#import required libraries
import pandas as pd
import numpy as np
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import itertools
from sklearn.feature_extraction.text import CountVectorizer


class Encoder:
    def __init__(self,data:list, verbose = False):
        """Initialize the Encoder class which stores data and applies various forms of encoding to it

        Args:
            data (list): a list of the data
            verbose (bool, optional): on True, prints out extra information. Defaults to False.
        """
        #variable to store the raw data
        self.data = data

        #variable to store the encoding keys
        self.encoding_mappings = {}

        #variable to store the encoded data
        self.encoded_data: np.array = []

        #configure a tokenizer, lematizer, and vectorizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lematizer = WordNetLemmatizer()
        self.vectorizer: CountVectorizer

        #for bag of words, store a list of the words in the bag
        self.words = []

        #set the encoding method to None for now
        self.encoding_method = None

        #set verbose status
        self.verbose = verbose

        return


    def encode(self,encoding_mappings:dict = None, normalize = True):
        """Encodes the data from a list of strings to a list of numbers

        Args:
            encoding_keys (dict, optional): Optional ability to apply a custom set of encoding keys If none is provided, then the function generates them automatically. Defaults to None.
            normalize (bool, optional): normalize the encoded values to be between 0 and 1. Defaults to True.
        """
        #set the encoding method
        self.encoding_method = "encode"
        
        #generate feature,encoding pairs for each possible feature in the data
        if not encoding_mappings:
            #get a list of unique strings in the data
            features = list(set(self.data))
            num_features = len(features)
            encodings = [i + 1 for i in range(num_features)]
            self.encoding_mappings = {feature:encoding for feature,encoding in zip(features,encodings)}
        else:
            self.encoding_mappings = encoding_mappings
        
        #apply normalization if desired
        if normalize:
            #get the max and min value
            max_val = max(self.encoding_mappings.values())
            min_val = min(self.encoding_mappings.values())
            for key in self.encoding_mappings:
                self.encoding_mappings[key] = (self.encoding_mappings[key] - min_val)/(max_val - min_val)
        
        #apply the encoding
        self.encoded_data = np.array([self.encoding_mappings[feature] for feature in self.data])

        if self.verbose:
            print("encoding_mappings: {}\n".format(self.encoding_mappings))
        
        return


    def bag_of_words(self, clean_strings = True, remove_stop_words = True, lematize = True):
        """Performs bag-of-words tokenization on the data

        Args:
            clean_strings (bool, optional): On True, will clean the data strings to be more uniform. Defaults to True.
            remove_stop_words (bool, optional): On true, will remove very common words. Defaults to True.
            lematize (bool, optional): On true, will lematize words. Defaults to True.
        """
        #set encoding method
        self.encoding_method = "bag-of-words"

        #clean the text data
        if clean_strings:
            self.data = [
                self.clean_text(
                text,
                remove_stop_words=remove_stop_words,
                lematize=lematize) for text in self.data]
        
        #vectorize the strings
        self.vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=lambda doc: doc,
            ngram_range=(1,1),
            analyzer='word'
        )
        self.encoded_data = self.vectorizer.fit_transform(self.data).toarray()
        
        #get a list of words
        self.words = self.vectorizer.get_feature_names_out()
        num_unique_words = len(self.words)
        encodings = [i  for i in range(num_unique_words)]
        self.encoding_mappings = {word:encoding for word,encoding in zip(self.words,encodings)}
        
        #reset the vectorizer to use only this set of words moving forward
        self.vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=lambda doc: doc,
            ngram_range=(1,1),
            analyzer='word',
            vocabulary= self.encoding_mappings
        )
        self.encoded_data = self.vectorizer.fit_transform(self.data).toarray()

        return
    
    def clean_text(self,text, remove_stop_words = True, lematize = True):
        """removes common words, converts to lower case, and decontracts a given string

        Args:
            text (_type_): a string of text to clean
            remove_stop_words (bool, optional): removes common words on true. Defaults to True.
            lematize (bool, optional): lematizes words on True. Defaults to True.

        Returns:
            _type_: _description_
        """
        # Convert words to lower case
        text = text.lower()

        # Expand contractions
        text  =  self.decontracted(text)
        # remove numbers
        text = re.sub(r"\d+",'',text)
        # substitue U.S. with united states
        text = re.sub(r"u.s.",' united states ',text)

        #remove punctuation and tokenize
        text = self.tokenizer.tokenize(text)

        #remove stop words
        if remove_stop_words:
            stop_words = set(stopwords.words("english"))
            text = [w for w in text if not w in stop_words]
        
        if lematize:
            text = list(map(lambda word: self.lematizer.lemmatize(word), text))
        
        if "stateselections" in text:
            a = 10

        return text
    
    def decontracted(self,phrase):
        """Decontracts a phrase on true

        Args:
            phrase (_type_): the phrase to be decontracted

        Returns:
            _type_: the decontracted string
        """
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def get_most_common(self,max_terms = 10):

        #ensure that we aren't requesting more terms than available
        n = max_terms
        if max_terms > np.shape(self.encoded_data)[1]:
            n = len(term_frequency)

        #compute the number of times each term appears
        term_frequency = np.sum(self.encoded_data,0)

        #get the arg sort to find the indicies of the terms that appear the most
        sorted_indicies = np.argsort(-1 * term_frequency)

        #get the terms that appear the most
        most_common_features = []
        for i in range(0,n):
            idx = sorted_indicies[i]
            feature = self.words[idx]
            most_common_features.append(feature)

        #get their respective count
        counts = term_frequency[sorted_indicies]

        #return the terms and their respective count
        
        
        return most_common_features,counts[0:n]
    
    def apply_encoding_to_new_data(self,new_data:list):
        pass
