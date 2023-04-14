#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from CustomExceptions import IncompatibleEncoder

#import the encoder class
from Encoder import Encoder

class PreProcessor:

    def __init__(self, verbose = False):
        """Create a new PreProcessor Class

        Args:
            verbose (bool, optional): _description_. Defaults to False.
        """

        self.df: pd.DataFrame
        self.headers = []
        
        #parameters to store the label header title and values
        self.label_title = ""
        self.label_encoder:Encoder

        self.verbose = verbose

    def import_data_from_file(self, file_name: str,deliminator=',',headers: list = [], replace_Null_NaN = True):
        """Import data from a file into the PreProcessor

        Args:
            file_name (str): filename of the dataset in the "datasets" folder (must be in this folder)
            deliminator (str, optional): deliminator used in the data file. Defaults to ','.
            headers (list, optional): the headers used in the file. Defaults to [].
            replace_Null_NaN (bool, optional): replaces Nan and Null values in the data set with "N/A" . Defaults to True.
        """
        
        #locate the file
        root_folder = os.path.dirname(os.getcwd())
        datasets_folder = os.path.join(root_folder,'datasets/')
        path = "".join([datasets_folder,file_name])
        
        #create the dataframe
        self.df = pd.read_csv(path,delimiter=deliminator,dtype = object)

        if replace_Null_NaN:
            self.df.fillna("N/A",inplace=True)

        if headers:
            self.headers = headers
            self.df.columns = headers
        
        if self.verbose:
            print("PreProcessor.__init()__: Data Imported")
        
        return
    
    def set_label_header(self,label_header = "label", encoding_mapping:dict = None, normalize = False, binarize = False):
        """Sets which column in the dataframe corresponds to the label for each data sample

        Args:
            label_header (str, optional): Column title corresponding to the labels. Defaults to "label".
            encoding_mapping (dict, optional): A dictionary of key:value pairs to use to encode the data. Defaults to None.
            normalize (bool, optional): normalizes the encodings to be between 0 and 1. Defaults to False.
            binarize (bool, optional): normalizes, then binarizes the labels to be either 0 or 1 (use only if supplying a known mapping that makes sense). Defaults to False.
        """

        #set the label headers
        self.label_title = label_header

        #get the labels and store them in an array
        self.label_encoder = Encoder(self.df[label_header].to_list(),verbose=self.verbose)
        self.label_encoder.encode(encoding_mapping,normalize,binarize)
        
        return
    
    def get_bag_of_words_encoder_for_feature(self,feature_name,clean_strings = True, remove_stop_words = True, lematize = True):
        """Returns a bag-of-words Encoder

        Args:
            feature_name (_type_): _description_
            clean_strings (bool, optional): Clean the strings in the data on True. Defaults to True.
            remove_stop_words (bool, optional): Remove very common words from each string on True. Defaults to True.
            lematize (bool, optional): Reduce words down to their base word on True. Defaults to True.

        Returns:
            Encoder: Encoder that has processed the data for bag-of-words
        """
        
        feature_encoder = Encoder(self.df[feature_name],verbose=self.verbose)
        feature_encoder.bag_of_words(
            clean_strings=clean_strings,
            remove_stop_words=remove_stop_words,
            lematize=lematize
        )
        return feature_encoder
    
    def get_most_common_words(self,encoder:Encoder, max_terms = 10,label_filters:list = None):
        """Returns the most common words and their counts for a given bag of words encoding

        Args:
            max_terms (int, optional): The number of words and counts to give. Defaults to 10.

        Returns:
            _type_: _description_
        """

        #ensure that the encoder is a bag-of-words encoder
        if encoder.encoder_type != 'bag-of-words':
            raise IncompatibleEncoder('bag-of-words',encoder)

        #ensure that we aren't requesting more terms than available
        n = max_terms
        if max_terms > np.shape(encoder.encoded_data)[1]:
            n = np.shape(encoder.encoded_data,1)[1]

        #compute the number of times each term appears
        term_frequency = np.sum(encoder.encoded_data,0)

        #get the arg sort to find the indicies of the terms that appear the most
        sorted_indicies = np.argsort(-1 * term_frequency)

        #get the terms that appear the most
        most_common_features = []
        for i in range(0,n):
            idx = sorted_indicies[i]
            feature = encoder.words[idx]
            most_common_features.append(feature)

        #get their respective count
        counts = term_frequency[sorted_indicies]

        #return the terms and their respective count
        return most_common_features,counts[0:n]