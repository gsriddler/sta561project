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

        #enabling the ability filter encoding to only specific words in the bag-of-words
        #TODO: Implement Filtering for specific features encodings/words in the samples
        self.filtering_enabled = False
        self.filtered_encoding_mappings = {}
        self.filtered_encoded_data: np.array = []

        self.filtered_words = [] #specific to bag of words filtering
        
        #set the encoding type to None for now
        self.encoder_type = None

        #set verbose status
        self.verbose = verbose

        return


    def encode(self,encoding_mappings:dict = None, normalize = True, Binarize = False):
        """Encodes the data from a list of strings to a list of numbers

        Args:
            encoding_keys (dict, optional): Optional ability to apply a custom set of encoding keys If none is provided, then the function generates them automatically. Defaults to None.
            normalize (bool, optional): normalize the encoded values to be between 0 and 1. Defaults to True.
            Binarize (bool, optional): on True, will normalize the encoded values and then set to either 0 or 1. Defaults to False.
        """
        #set the encoding method
        self.encoder_type = "encode"
        
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
        if normalize or Binarize:
            #get the max and min value
            max_val = max(self.encoding_mappings.values())
            min_val = min(self.encoding_mappings.values())
            for key in self.encoding_mappings:
                self.encoding_mappings[key] = (self.encoding_mappings[key] - min_val)/(max_val - min_val)
        
        #apply the encoding
        self.encoded_data = np.array([self.encoding_mappings[feature] for feature in self.data])

        if Binarize:
            for key in self.encoding_mappings:
                self.encoding_mappings[key] = round(self.encoding_mappings[key])

        if self.verbose:
            print("Encoder.encode: encoding_mappings: {}\n".format(self.encoding_mappings))
        
        return


    def bag_of_words(self, clean_strings = True, remove_stop_words = True, lematize = True):
        """Performs bag-of-words tokenization on the data

        Args:
            clean_strings (bool, optional): On True, will clean the data strings to be more uniform. Defaults to True.
            remove_stop_words (bool, optional): On true, will remove very common words. Defaults to True.
            lematize (bool, optional): On true, will lematize words. Defaults to True.
        """
        #set encoding method
        self.encoder_type = "bag-of-words"

        #clean the text data
        if clean_strings:
            self.data = [
                self._clean_text(
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
    
    def _clean_text(self,text, remove_stop_words = True, lematize = True):
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
        text  =  self._decontracted(text)
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
    
    def _decontracted(self,phrase):
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

    def apply_filter(self, filtered_terms:list):

        #set filtering enabled flag
        self.filtering_enabled = True

        #TODO: implement filtering
    
    def apply_encoding_to_new_data(self,new_data:list):
        pass
