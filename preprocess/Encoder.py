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
from CustomExceptions import IncompatibleEncoder,FilteringNotEnabled


class Encoder:
    def __init__(self,data:list, verbose = False):
        """Initialize the Encoder class which stores data and applies various forms of encoding to it

        Args:
            data (list): a list of the data
            verbose (bool, optional): on True, prints out extra information. Defaults to False.
        """
        #variable to store the raw data
        self.data:pd.DataFrame = data

        #variable to store the encoding keys
        self.encoding_mappings = {}

        #variable to store the feature names
        self.feature_names = []

        #variable to store the encoded data
        self.encoded_data: np.array = []


        #Encoder Relevant Terms
        self.normalize:bool #also used for weighted-average encoder
        self.binarize:bool

        #Bag-of-words relevant terms
        #configure a tokenizer, lematizer, and vectorizer
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lematizer = WordNetLemmatizer()
        self.vectorizer: CountVectorizer
        self.clean_strings = True
        self.remove_stop_words = True
        self.lematize = True
        
        #Credit History Variables
        self.credit_weights = np.array([[-1,-0.5,0,0.5,1]])
        self.compute_credit_history_enable = False

        #enabling the ability filter encoding to only specific words in the bag-of-words
        #TODO: Implement Filtering for specific features encodings/words in the samples
        self.filtering_enabled = False
        self.filtered_encoding_mappings = {}
        self.filtered_encoded_data: np.array = []
        self.filtered_feature_names = []

        #new vectorizer specific to filtering
        self.filtered_vectorizer: CountVectorizer
        
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

        #set normalization and binarization status
        self.normalize = normalize
        self.binarize = Binarize
        
        #generate feature,encoding pairs for each possible feature in the data
        if not encoding_mappings:
            #get a list of unique strings in the data
            self.feature_names = list(set(self.data))
            num_features = len(self.feature_names)
            encodings = [i + 1 for i in range(num_features)]
            self.encoding_mappings = {feature:encoding for feature,encoding in zip(self.feature_names,encodings)}
        else:
            self.encoding_mappings = encoding_mappings.copy()
            self.feature_names = [feature for feature in encoding_mappings.keys()]
        
        #apply normalization if desired
        if normalize or Binarize:
            #get the max and min value
            max_val = max(self.encoding_mappings.values())
            min_val = min(self.encoding_mappings.values())
            for key in self.encoding_mappings:
                self.encoding_mappings[key] = (self.encoding_mappings[key] - min_val)/(max_val - min_val)
        
        #binarize the encodings if desired
        if Binarize:
            for key in self.encoding_mappings:
                self.encoding_mappings[key] = round(self.encoding_mappings[key])

        #apply the encoding
        self.encoded_data = self._apply_encode_encoding(self.data,apply_filters=False)

        if self.verbose:
            print("Encoder.encode: encoding_mappings: {}\n".format(self.encoding_mappings))
        
        return
    
    def _apply_encode_encoding(self,data:pd.Series, apply_filters = False):
        """Apply the bag-of-words encoding to the given data

        Args:
            data (pd.Series): the data to apply the encoding to
            apply_filters (bool, optional): On True, applies filtering to the encoded data. Defaults to False.

        Raises:
            FilteringNotEnabled: Raised when filters are requested to be applied, but filtering is not enabled

        Returns:
            np.array: the encoded data
        """
        
        if (apply_filters):
            if self.filtering_enabled:
                #return the filtered-encoded data
                return np.transpose(np.array([[self.filtered_encoding_mappings[feature] 
                                if feature in self.filtered_feature_names
                                else self.filtered_encoding_mappings["other"] 
                                for feature in data]]))
            else:
                raise FilteringNotEnabled(self.filtering_enabled)
        else:
            return np.transpose(np.array([[self.encoding_mappings[feature] for feature in data]]))


    def bag_of_words(self, clean_strings = True, remove_stop_words = True, lematize = True):
        """Performs bag-of-words tokenization on the data

        Args:
            clean_strings (bool, optional): On True, will clean the data strings to be more uniform. Defaults to True.
            remove_stop_words (bool, optional): On true, will remove very common words. Defaults to True.
            lematize (bool, optional): On true, will lematize words. Defaults to True.
        """
        #set encoding method
        self.encoder_type = "bag-of-words"

        #save encoding settings
        self.clean_strings = clean_strings
        self.remove_stop_words = remove_stop_words
        self.lematize = lematize

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
        self.feature_names = self.vectorizer.get_feature_names_out()
        num_unique_words = len(self.feature_names)
        encodings = [i  for i in range(num_unique_words)]
        self.encoding_mappings = {word:encoding for word,encoding in zip(self.feature_names,encodings)}
        
        #reset the vectorizer to use only this set of words moving forward
        self.vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=lambda doc: doc,
            ngram_range=(1,1),
            analyzer='word',
            vocabulary= self.encoding_mappings
        )

        #apply the encoding
        self.encoded_data = self._apply_bag_of_words_encoding(self.data,apply_filters=False)

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

    def _apply_bag_of_words_encoding(self,data: pd.Series,apply_filters = False):
        """Apply the bag-of-words encoding to the given data

        Args:
            data (pd.Series): the data to apply the encoding to
            apply_filters (bool, optional): On True, applies filtering to the encoded data. Defaults to False.

        Raises:
            FilteringNotEnabled: Raised when filters are requested to be applied, but filtering is not enabled

        Returns:
            np.array: the encoded data
        """

        if (apply_filters):
            if self.filtering_enabled:
                return self.filtered_vectorizer.fit_transform(data).toarray()
            else:
                raise FilteringNotEnabled(self.filtering_enabled)
        else:
            return self.vectorizer.fit_transform(data).toarray()

    def credit_history(self,compute_credit_history=True):
        """Compute the credit history for each row of the data

        Args:
            compute_credit_history(bool, optional): on True, computes a single value using a weighted average to represent the credit history. On false, leaves the creit history simply as an array value. Defaults to True
        """
        #set the encoder type
        self.encoder_type = "credit history"
        self.compute_credit_history_enable = compute_credit_history

        if compute_credit_history:
            #set the feature names
            self.feature_names = ["credit history"]
        else:
            self.feature_names = ['count_1','count_2','count_3','count_4','count_5']
        
        self.encoded_data = self._apply_credit_history_encoding(self.data)
    
    def _apply_credit_history_encoding(self,data:pd.DataFrame):
        """Apply credit history encoding to the given pandas Data Frame

        Args:
            data (pd.DataFrame): the data to apply the encoding to

        Returns:
            np.array: the encoded data
        """

        #convert the data to be a numpy array
        credit_counts = data.replace('N/A','0').astype('int').to_numpy()

        if self.compute_credit_history_enable:
            return np.transpose(np.array([self._compute_credit_histories(credit_counts)]))
        else:
            return np.copy(credit_counts)


    def _compute_credit_histories(self,credit_counts:np.array):
        """Compute the credit history score

        Args:
            credit_counts (np.array): list of credit counts in the order of ['pants on fire','false','barely true','half true',mostly true']
        """
        
        sums = np.sum(credit_counts,1)
        weighted_credit_counts = credit_counts / sums[:,None]
        #set any nan values to zero as there isn't any credit associated with it
        weighted_credit_counts[np.isnan(weighted_credit_counts)] = 0
        return np.matmul(weighted_credit_counts,np.transpose(self.credit_weights))[:,0]
    
    def configure_filter(self, filtered_terms:list):

        #set filtering enabled flag
        self.filtering_enabled = True

        self.filtered_encoding_mappings = {}
        self.filtered_encoded_data: np.array = []

        #configure the requested filter
        if self.encoder_type == "encode":
            self._configure_filter_encode(filtered_terms)
        elif self.encoder_type == "bag-of-words":
            self._configure_filter_bag_of_words(filtered_terms)
        else:
            raise IncompatibleEncoder("encode or bag-of-words",self.encoder_type)


    def _configure_filter_bag_of_words(self, filtered_terms:list):
        """Configure and apply  a filter to data that has been encoded ot be a bag-of-words. 
        The filter reduces the number of features to only those included in the provided filtered_terms list.
        Input samples that do not include any of the filtered terms will have all zeros

        Args:
            filtered_terms (list): A list of the only terms that should be used to encode the data
        """

        #set the filtered words list
        self.filtered_feature_names = filtered_terms

        num_filtered_words = len(self.filtered_feature_names)
        encodings = [i  for i in range(num_filtered_words)]
        self.filtered_encoding_mappings = {word:encoding for word,encoding in zip(self.filtered_feature_names,encodings)}
        
        #reset the vectorizer to use only this set of words moving forward
        self.filtered_vectorizer = CountVectorizer(
            lowercase=False,
            tokenizer=lambda doc: doc,
            ngram_range=(1,1),
            analyzer='word',
            vocabulary= self.encoding_mappings
        )
        self.filtered_encoded_data = self._apply_bag_of_words_encoding(self.data,apply_filters=True)

        return

    def _configure_filter_encode(self, filtered_terms: list):
        """Configure and apply  a filter to data that has been encoded using the encode method of encoding. 
        The filter reduces the number of features to only those included in the provided filtered_terms list.
        Input samples that do not include any of the filtered terms will have all zeros

        Args:
            filtered_terms (list): A list of the only terms that should be used to encode the data
        """
        #setup the feature names, include and include an encoding for "other" in the encodings
        self.filtered_feature_names = ["other"]
        for item in filtered_terms:
            self.filtered_feature_names.append(item)
        
        #generate a new series of filtered feature encodings
        num_features = len(self.feature_names)
        encodings = [i for i in range(num_features)]
        self.filtered_encoding_mappings = {feature:encoding for feature,encoding in zip(self.filtered_feature_names,encodings)}

        #apply normalization if desired
        if self.normalize or self.binarize:
            #get the max and min value
            max_val = max(self.filtered_encoding_mappings.values())
            min_val = min(self.filtered_encoding_mappings.values())
            for key in self.filtered_encoding_mappings:
                self.filtered_encoding_mappings[key] = (self.filtered_encoding_mappings[key] - min_val)/(max_val - min_val)
        
        if self.binarize:
            for key in self.filtered_encoding_mappings:
                self.filtered_encoding_mappings[key] = round(self.filtered_encoding_mappings[key])
        
        #apply the encoding
        self.filtered_encoded_data = self._apply_encode_encoding(self.data,apply_filters=True)

        return
    

    
    def apply_encoding_to_new_data(self,new_data:pd.DataFrame, apply_filter_if_enabled = True):
        """Apply the encoding on a new set of data

        Args:
            new_data (pd.DataFrame): the new data that the encoding should be applied to
            apply_filter_if_enabled (bool, optional): On true, applies filtering to the data if it is enabled (only on encode or bag-of-words encoding). Defaults to True.

        Raises:
            IncompatibleEncoder: _description_

        Returns:
            _type_: _description_
        """
        
        #determine if filtering should be applied
        if (apply_filter_if_enabled and self.filtering_enabled):
            apply_filtering = True
        else:
            apply_filtering = False
        
        if self.encoder_type == "encode":
            return self._apply_encode_encoding(new_data,apply_filters=apply_filtering)
        elif self.encoder_type == "bag-of-words":
            #perform any clearning of the data
            if self.clean_strings:
                new_data = [
                    self._clean_text(
                    text,
                    remove_stop_words=self.remove_stop_words,
                    lematize=self.lematize) for text in new_data]
            return self._apply_bag_of_words_encoding(new_data, apply_filters=apply_filtering)
        elif self.encoder_type == "credit history":
            return self._apply_credit_history_encoding(new_data)
        else:
            raise IncompatibleEncoder("encode,bag-of-words,credit history",self.encoder_type)

    def get_encoded_data(self):

        if self.filtering_enabled:
            return self.filtered_encoded_data
        else:
            return self.encoded_data
    
    def get_feature_names(self):
        if self.filtering_enabled:
            return self.filtered_feature_names
        else:
            return self.feature_names