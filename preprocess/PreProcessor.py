#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    def set_label_header(self,label_header = "label", encoding_mapping = None, normalize_encoding = False):
        

        #set the label headers
        self.label_title = label_header

        #get the labels and store them in an array
        self.label_encoder = Encoder(self.df[label_header].to_list(),verbose=self.verbose)
        self.label_encoder.encode(encoding_mapping,normalize_encoding)
        
        return
    
    def get_bag_of_words_encoder_for_feature(self,feature_name,clean_strings = True, remove_stop_words = True, lematize = True):
        
        feature_encoder = Encoder(self.df[feature_name],verbose=self.verbose)
        feature_encoder.bag_of_words(
            clean_strings=clean_strings,
            remove_stop_words=remove_stop_words,
            lematize=lematize
        )
        return feature_encoder