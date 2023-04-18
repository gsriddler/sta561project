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
    
    def get_standard_encoder_for_features(self,feature_name,encoding_mapping:dict = None,normalize = False,Binarize = False):
        """Returns a standard Encoder (creates Encoder of type 'encode')

        Args:
            encoding_mapping (dict, optional): A dictionary of key:value pairs to use to encode the data. Defaults to None.
            normalize (bool, optional): normalizes the encodings to be between 0 and 1. Defaults to False.
            binarize (bool, optional): normalizes, then binarizes the encodings to be either 0 or 1 (use only if supplying a known mapping that makes sense). Defaults to False.
        """
        feature_encoder = Encoder(self.df[feature_name],verbose=self.verbose)
        feature_encoder.encode(
            encoding_mappings=encoding_mapping,
            normalize=normalize,
            Binarize=Binarize
        )

        return feature_encoder
    
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
    
    def get_most_popular_features(self,encoder:Encoder,max_terms = 10,label_filters:list = None):
        """Returns the most popular features for a given encoder with the option to filter for specific labels

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify the most common features for
            max_terms (int, optional): the number of popular labels to identify. Defaults to 10.
            label_filters (list, optional): A list of desired labels to filter for. Defaults to None.

        Raises:
            IncompatibleEncoder: Encoder must be of type "encode" or "bag-of-words"

        Returns:
            list,nd.array: list of the most common features (sorted in order) and their respective counts
        """

        #filter the data for certain labels if requested
        if label_filters:
            #identify the indicies of the in the labels that match the filter constraints
            data = encoder.encoded_data[self._get_filter_indicies(label_filters)]
        else:
            data = encoder.encoded_data

        #ensure that we aren't requesting more terms than available
        n = max_terms
        if max_terms > len(encoder.feature_names):
            n = len(encoder.feature_names)
        
        if encoder.encoder_type == 'encode':
            term_frequency = np.array([np.count_nonzero(data == encoder.encoding_mappings[feature]) for feature in encoder.feature_names])
        elif encoder.encoder_type == 'bag-of-words':
            term_frequency = np.sum(data,0)
        else:
            raise IncompatibleEncoder('encode or bag-of-words',encoder)
    
        #get the arg sort to find the indicies of the terms that appear the most
        sorted_indicies = np.argsort(-1 * term_frequency)

        #get the terms that appear the most
        most_common_features = []
        for i in range(0,n):
            idx = sorted_indicies[i]
            feature = encoder.feature_names[idx]
            most_common_features.append(feature)

        #get their respective count
        counts = term_frequency[sorted_indicies]

        #return the terms and their respective count
        return most_common_features,counts[0:n]
    
    def plot_most_popular_features(self,encoder:Encoder,labels_to_plot:list = None, max_terms = 10):
        """Plots a sub-plot of the max_terms most popular features for each label in the labels_to_plot list

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify the most common features for
            labels_to_plot (list,optional): The labels to generate subplots for. When none, generates a subplot for every label Defaults to None
            max_terms (int, optional): The number of terms to plot for each label. Defaults to 10.
        """

        if not labels_to_plot:
            labels_to_plot = self.label_encoder.feature_names

        fig,axs = plt.subplots(len(labels_to_plot),figsize=(1.5 * max_terms,2 * len(labels_to_plot)),gridspec_kw={'hspace':0.75})

        for i in range(len(labels_to_plot)):

            label = labels_to_plot[i]
            most_common_features,counts = self.get_most_popular_features(
                encoder=encoder,
                max_terms=max_terms,
                label_filters=[label]
            )

            axs[i].bar(most_common_features,counts)
            axs[i].set_title("Most popular features for posts labeled: {}".format(label))
            axs[i].set_xlabel("Most popular features")
            axs[i].set_ylabel("Count")

        plt.show()

        return

    def plot_count_for_features(self,encoder:Encoder,features_to_plot:list, labels_to_plot:list = None):
        """Plots a line for the number of times each figure appears for each label that is requested

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify number of times specific features appear for
            features_to_plot (list,optional): The features that we wish to determine the count for
            labels_to_plot (list,optional): The labels to generate subplots for. When none, generates a subplot for every label Defaults to None
        """
        
        #compute the counts corresponding to each feature and each label

        if not labels_to_plot:
            labels_to_plot = self.label_encoder.feature_names

        counts = np.zeros((len(labels_to_plot),len(features_to_plot)))

        for i in range(len(labels_to_plot)):
            label = labels_to_plot[i]
            #get the data corresponding to the label
            data = encoder.encoded_data[self._get_filter_indicies([label])]

            for j in range(len(features_to_plot)):
                feature = features_to_plot[j]
                if encoder.encoder_type == 'encode':
                    count = np.count_nonzero(data == encoder.encoding_mappings[feature])
                elif encoder.encoder_type == 'bag-of-words':
                    count = np.sum(data,0)[encoder.encoding_mappings[feature]]
                else:
                    raise IncompatibleEncoder('encode or bag-of-words',encoder)
                counts[i,j] = count

        #generate the plot
        plt.figure(figsize=(1 * len(labels_to_plot),4))
        for i in range(len(features_to_plot)):
            plt.plot(labels_to_plot,counts[:,i],label=features_to_plot[i])
        
        plt.title("Feature Count vs Sample Label")
        plt.xlabel("Labels")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

        return


        


    def _get_filter_indicies(self,label_filters:list):
        """Get an array of True/False values for samples that have the desired labels

        Args:
            label_filters (list): A list of labels for which to filter for

        Returns:
            np.ndarray: an array of True/False values for samples that have the desired labels
        """

        #identify the indicies of the labels that match the filter constraints
        valid_indicies = np.zeros(np.shape(self.label_encoder.encoded_data)[0])

        labels = self.label_encoder.encoded_data
        label_encodings = self.label_encoder.encoding_mappings

        for label in label_filters:
            valid_indicies = np.logical_or(valid_indicies,labels == label_encodings[label])
        
        return valid_indicies