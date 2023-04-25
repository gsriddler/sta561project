#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#import the encoder class
from Encoder import Encoder

#import the preprocessor class
from PreProcessor import PreProcessor

from CustomExceptions import IncompatibleEncoder

class LiarLiarPreProcessor:

    def __init__(self, verbose = False):

        #set the verbose settings
        self.verbose = verbose

        #configure the labels in the dataset
        self.headers = []

        #configuration parameters
        self.encoder_parameters = []
        
        #list of the encoders being used
        self.encoders = []
        
        #configure the pre-processor
        self.pre_processor = PreProcessor(verbose=verbose)

        #import settings
        self.deliminator = ""
        self.replace_Null_NaN = True
    
    def import_training_data(self, 
            file_name = "train.tsv",
            deliminator = '\t' ,
            custom_headers:list = None, 
            replace_Null_NaN = True):
        """Imports data for the training dataset

        Args:
            file_name (str): filename of the dataset in the "datasets" folder (must be in this folder)
            deliminator (str, optional): deliminator used in the data file. Defaults to ','.
            custom_headers (list, optional): the headers used in the file. Defaults to [].
            replace_Null_NaN (bool, optional): replaces Nan and Null values in the data set with "N/A" . Defaults to True.
        """

        #set the data labels
        if custom_headers:
            self.headers = custom_headers
        else:
            self.headers = [
                'id',                # Column 1: the ID of the statement ([ID].json).
                'label',             # Column 2: the label.
                'statement',         # Column 3: the statement.
                'subjects',          # Column 4: the subject(s).
                'speaker',           # Column 5: the speaker.
                'speaker_job_title', # Column 6: the speaker's job title.
                'state_info',        # Column 7: the state info.
                'party_affiliation', # Column 8: the party affiliation.
                
                # Column 9-13: the total credit history count, including the current statement.
                'count_1', # pants on fire counts.
                'count_2', # false counts.
                'count_3', # barely true counts.
                'count_4', # half true counts.
                'count_5', # mostly on fire counts.
                
                'context' # Column 14: the context (venue / location of the speech or statement).
            ]
        
        self.pre_processor.import_data_from_file(
            file_name=file_name,
            deliminator=deliminator,
            headers=self.headers,
            replace_Null_NaN=replace_Null_NaN
        )

        self.deliminator = deliminator
        self.replace_Null_NaN = replace_Null_NaN

        return
    
    def set_label_header(self,
            label_header='label',
            custom_label_encoding=None,
            normalize=False,
            binarize=False):
        """_summary_

        Args:
            label_header (str, optional): Column title corresponding to the labels. Defaults to "label".
            custom_label_encoding (dict, optional): A dictionary of key:value pairs to use to encode the data. Defaults to None.
            normalize (bool, optional): normalizes the encodings to be between 0 and 1. Defaults to False.
            binarize (bool, optional): normalizes, then binarizes the labels to be either 0 or 1 (use only if supplying a known mapping that makes sense). Defaults to False.
        """
        
        #set the label mapping
        if custom_label_encoding:
            label_mapping = custom_label_encoding
        else:
            label_mapping = {'pants-fire':0,
             'false':1,
             'barely-true':2,
             'half-true':3,
             'mostly-true':4,
             'true':5}
        
        #set the label mapping in the preprocessor
        self.pre_processor.set_label_header(
            label_header='label',
            encoding_mapping=label_mapping,
            normalize=normalize,
            binarize=binarize
        )

        return
    

    def configure_encodings(self,encoder_parameters:list):
        """Configure the encodings to be used to generate the dataset

        Args:
            encoder_parameters (list): a list of encoder configuration dictionaries
        """

        #save the encoder parameters to a list
        self.encoder_parameters = encoder_parameters

        #reset the encoders list
        self.encoders = []
        
        #initialize an encoder with the data for each encoder parameter dictionary in the encoder_parameters list
        for configuration in encoder_parameters:
            self.add_encoding(configuration)
        
        return

    def add_encoding(self,encoder_configuration:dict):
        """Add an encoding to the self.encoders list, type of encoder is automatically loaded

        Args:
            encoder_configuration (dict): dictionary containing key encoder configuration parameters

        Raises:
            IncompatibleEncoder: _description_
        """

        encoder_type = encoder_configuration["encoder_type"]

        if encoder_type == "encode":
            self.encoders.append(self._get_standard_encoder(encoder_configuration))
        elif encoder_type == "bag-of-words":
            self.encoders.append(self._get_bag_of_words_encoder(encoder_configuration))
        elif encoder_type == "credit history":
            self.encoders.append(self._get_credit_score_encoder(encoder_configuration))
        else:
            raise IncompatibleEncoder("encode,bag-of-words,or credit history",encoder_type)
        
        return

    def _get_standard_encoder(self,encoder_configuration:dict):
        """Return a standard encoder 

        Args:
            encoder_configuration (dict): dictionary containing key encoder configuration parameters

        Returns:
            Encoder: standard
        """
        
        #get the feature name
        feature_name = encoder_configuration["feature_name"]

        #get the encoding mapping
        if "encoding_mapping" in encoder_configuration.keys():
            encoding_mapping = encoder_configuration["encoding_mapping"]
        else:
            encoding_mapping = None
        
        #normalization setting
        if "normalize" in encoder_configuration.keys():
            normalize = encoder_configuration["normalize"]
        else:
            normalize = False

        #Binarize
        if "Binarize" in encoder_configuration.keys():
            Binarize = encoder_configuration["Binarize"]
        else:
            Binarize = False


        encoder = self.pre_processor.get_standard_encoder_for_features(
            feature_name=feature_name,
            encoding_mapping=encoding_mapping,
            normalize=normalize,
            Binarize=Binarize
        )

        #configure filtering if enabled
        if "filtering" in encoder_configuration.keys():
            if "filtering_enabled" in encoder_configuration["filtering"].keys():
                filtering_enabled = encoder_configuration["filtering"]["filtering_enabled"]
                if filtering_enabled:
                    encoder.configure_filter(encoder_configuration["filtering"]["filtered_terms"])
        
        return encoder
    
    def _get_bag_of_words_encoder(self,encoder_configuration:dict):
        """Return a bag-of-words encoder 

        Args:
            encoder_configuration (dict): dictionary containing key encoder configuration parameters

        Returns:
            Encoder: bag-of-words encoder
        """
        
        #get the feature name
        feature_name = encoder_configuration["feature_name"]

        #get the encoding mapping
        if "clean_strings" in encoder_configuration.keys():
            clean_strings = encoder_configuration["clean_strings"]
        else:
            clean_strings = None
        
        #normalization setting
        if "remove_stop_words" in encoder_configuration.keys():
            remove_stop_words = encoder_configuration["remove_stop_words"]
        else:
            remove_stop_words = False

        #Binarize
        if "lematize" in encoder_configuration.keys():
            lematize = encoder_configuration["lematize"]
        else:
            lematize = False


        encoder = self.pre_processor.get_bag_of_words_encoder_for_feature(
            feature_name=feature_name,
            clean_strings=clean_strings,
            remove_stop_words=remove_stop_words,
            lematize=lematize
        )

        #configure filtering if enabled
        if "filtering" in encoder_configuration.keys():
            if "filtering_enabled" in encoder_configuration["filtering"].keys():
                filtering_enabled = encoder_configuration["filtering"]["filtering_enabled"]
                if filtering_enabled:
                    encoder.configure_filter(encoder_configuration["filtering"]["filtered_terms"])
        
        return encoder

    def _get_credit_score_encoder(self,encoder_configuration:dict):
        """Return a credit score encoder 

        Args:
            encoder_configuration (dict): dictionary containing key encoder configuration parameters

        Returns:
            Encoder: credit score encoder
        """
        
        #get the feature name
        feature_names = encoder_configuration["feature_names"]

        #get the encoding mapping
        if "compute_credit_history" in encoder_configuration.keys():
            compute_credit_history = encoder_configuration["compute_credit_history"]
        else:
            compute_credit_history = True


        encoder = self.pre_processor.get_credit_history_encoder_for_features(
            feature_names=feature_names,
            compute_credit_history=compute_credit_history
        )
        
        return encoder
    
    def get_dataset(self):

        #get the encoded samples
        
        #get the first set of encoded samples
        X = np.empty((np.shape(self.encoders[0].encoded_data)[0],0))
        X_headers = []
        for i in range(len(self.encoders)):
            encoder = self.encoders[i]
            encoded_data = encoder.get_encoded_data()
            X = np.concatenate((X,encoded_data),axis=1)
            if np.shape(encoded_data)[1] > 1:
                X_headers.extend(encoder.get_feature_names())
            else:
                X_headers.append(self.encoder_parameters[i]["encoder_name"])
        
        y = self.pre_processor.label_encoder.encoded_data

        return y,X,X_headers
    
    def apply_encodings_to_new_data(self, file_name:str):
        
        df = self._import_data_from_file(file_name)

        #get the first set of encoded samples
        X = np.empty((df.shape[0],0))
        for i in range(len(self.encoders)):
            encoder = self.encoders[i]
            if ((encoder.encoder_type) == "encode" or (encoder.encoder_type == "bag-of-words")):
                feature_name = self.encoder_parameters[i]["feature_name"]
                encoded_data = encoder.apply_encoding_to_new_data(df[feature_name])
            elif encoder.encoder_type == "credit history":
                feature_names = self.encoder_parameters[i]["feature_names"]
                encoded_data = encoder.apply_encoding_to_new_data(df[feature_names])
            X = np.concatenate((X,encoded_data),axis=1)
        
        y = self.pre_processor.label_encoder.apply_encoding_to_new_data(df[self.pre_processor.label_title])

        return y,X


        
    def _import_data_from_file(self, file_name: str):
        """Import data from a file

        Args:
            ile_name (str): filename of the dataset in the "datasets" folder (must be in this folder)

        Returns:
            pd.DataFrame: pandas dataframe with the data
        """
        
        #locate the file
        root_folder = os.path.dirname(os.getcwd())
        datasets_folder = os.path.join(root_folder,'datasets/')
        path = "".join([datasets_folder,file_name])
        
        #create the dataframe
        df = pd.read_csv(path,delimiter=self.deliminator,dtype = object)

        if self.replace_Null_NaN:
            df.fillna("N/A",inplace=True)

        df.columns = self.headers
        
        if self.verbose:
            print("LiarLiarPreProcessor._import_data_from_file: Data Imported")
        
        return df