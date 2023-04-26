#import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#import the custom exceptions for encoder errors
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

    def get_credit_history_encoder_for_features(self,feature_names,compute_credit_history=True):
        """Returns a credit_history Encoder object

        Args:
            feature_names (_type_): the column names in the data array used to store credit history values.
            compute_credit_history(bool, optional): on True, computes a single value using a weighted average to represent the credit history. On false, leaves the creit history simply as an array value. Defaults to True
        Returns:
            Encoder: a credit history encoder
        """

        credit_history_encoder = Encoder(self.df[feature_names],verbose=self.verbose)
        credit_history_encoder.credit_history(compute_credit_history)
        return credit_history_encoder

    
    def get_most_popular_features(self,encoder:Encoder,max_terms = 10,label_filters:list = None, percentage = False):
        """Returns the most popular features for a given encoder with the option to filter for specific labels

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify the most common features for
            max_terms (int, optional): the number of popular labels to identify. Defaults to 10.
            label_filters (list, optional): A list of desired labels to filter for. Defaults to None.
            percentage (bool, optional): Normalizes the frequencies and returns them as a percentage. Defaults to False.

        Raises:
            IncompatibleEncoder: Encoder must be of type "encode" or "bag-of-words"

        Returns:
            list,nd.array: list of the most common features (sorted in order) and their respective counts
        """

        #filter the data for certain labels if requested
        if not label_filters:
#            #identify the indicies of the in the labels that match the filter constraints
#            data = encoder.encoded_data[[self._get_filter_indicies(label_filters)]]
#        else:
            label_filters = self.label_encoder.feature_names

        #ensure that we aren't requesting more terms than available
        n = max_terms
        if max_terms > len(encoder.feature_names):
            n = len(encoder.feature_names)
        
        #get the term frequencies
        feature_frequencies = self._get_feature_frequencies(encoder,percentage)
        
        #filter only for the specific terms that we are interested in
        for i in range(len(self.label_encoder.feature_names)):
            if self.label_encoder.feature_names[i] not in label_filters:
                feature_frequencies[:,i] = 0
        
        feature_frequencies = np.sum(feature_frequencies,1)
    
        #get the arg sort to find the indicies of the terms that appear the most
        sorted_indicies = np.argsort(-1 * feature_frequencies)

        #get the terms that appear the most
        most_common_features = []
        for i in range(0,n):
            idx = sorted_indicies[i]
            feature = encoder.feature_names[idx]
            most_common_features.append(feature)

        #get their respective count
        counts = feature_frequencies[sorted_indicies]

        #return the terms and their respective count
        return most_common_features,counts[0:n]

    def plot_most_popular_features(self,encoder:Encoder,labels_to_plot:list = None, max_terms = 10,percentage=False):
        """Plots a sub-plot of the max_terms most popular features for each label in the labels_to_plot list

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify the most common features for
            labels_to_plot (list,optional): The labels to generate subplots for. When none, generates a subplot for every label Defaults to None
            max_terms (int, optional): The number of terms to plot for each label. Defaults to 10.
            percentage (bool, optional): Normalizes the frequencies and returns them as a percentage. Defaults to False.
        """

        if not labels_to_plot:
            labels_to_plot = self.label_encoder.feature_names

        fig,axs = plt.subplots(len(labels_to_plot),figsize=(1.5 * max_terms,2 * len(labels_to_plot)),gridspec_kw={'hspace':0.75})

        for i in range(len(labels_to_plot)):

            label = labels_to_plot[i]
            most_common_features,counts = self.get_most_popular_features(
                encoder=encoder,
                max_terms=max_terms,
                label_filters=[label],
                percentage=percentage
            )

            axs[i].bar(most_common_features,counts)
            axs[i].set_title("Most popular features for posts labeled: {}".format(label))
            axs[i].set_xlabel("Most popular features")
            if percentage:
                axs[i].set_ylabel("Percentage")
            else:
                axs[i].set_ylabel("Count")

        plt.show()

        return

    def plot_count_for_features(self,encoder:Encoder,features_to_plot:list, labels_to_plot:list = None, percentage=False):
        """Plots a line for the number of times each figure appears for each label that is requested

        Args:
            encoder (Encoder): An encoder corresponding to the category of data that we want to identify number of times specific features appear for
            features_to_plot (list): The features that we wish to determine the count for
            labels_to_plot (list,optional): The labels to generate subplots for. When none, generates a subplot for every label Defaults to None
            percentage (bool, optional): on True, normalizes the counts to be the percentage that each feature applies to each label
        """
        
        #compute the counts corresponding to each feature and each label

        if not labels_to_plot:
            labels_to_plot = self.label_encoder.feature_names

        counts = np.zeros((len(features_to_plot),len(labels_to_plot)))

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
                    raise IncompatibleEncoder('encode or bag-of-words',encoder.encoder_type)
                counts[j,i] = count
        
        #normalize if requested
        if percentage:
            sums = np.sum(counts,1)
            counts = counts / sums[:,None]

        #generate the plot
        plt.figure(figsize=(1 * len(labels_to_plot),4))
        for i in range(len(features_to_plot)):
            plt.plot(labels_to_plot,counts[i,:],label=features_to_plot[i])
        
        plt.title("Feature Count vs Sample Label")
        plt.xlabel("Labels")
        if percentage:
            plt.ylabel("Percentage")
        else:
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

        labels = self.label_encoder.encoded_data[:,0]
        label_encodings = self.label_encoder.encoding_mappings

        for label in label_filters:
            valid_indicies = np.logical_or(valid_indicies,labels == label_encodings[label])
        
        return valid_indicies
    
    def _get_feature_frequencies(self,encoder:Encoder,percentage = False):
        """Returns the frequency (expressed as a count or percentage) that each feature for an encoder corresponds to a specific set of labels

        Args:
            encoder (Encoder): The encoder used to encode the data
            percentage (bool, optional): Normalizes the frequencies and returns them as a percentage. Defaults to False.

        Raises:
            IncompatibleEncoder: Raised when the encoder type is not 'encode' or 'bag-of-words'

        Returns:
            np.array: matrix of feature frequency where the rows are the individual features and the columns are the term frequencies
        """
        

        labels = self.label_encoder.feature_names
        
        #initialize the term frequency label
        feature_frequencies = np.zeros((len(encoder.feature_names),len(labels)))

        for i in range(len(labels)):

            #get the data corresponding to that filter
            data = encoder.encoded_data[self._get_filter_indicies([labels[i]])]

            #get the term frequency information
            if encoder.encoder_type == 'encode':
                feature_frequencies[:,i] = np.array([np.count_nonzero(data == encoder.encoding_mappings[feature]) for feature in encoder.feature_names])
            elif encoder.encoder_type == 'bag-of-words':
                feature_frequencies[:,i] = np.sum(data,0)
            else:
                raise IncompatibleEncoder('encode or bag-of-words',encoder.encoder_type)
        
        
        #apply normalization if requested
        if percentage:
            sums = np.sum(feature_frequencies,1)
            feature_frequencies = feature_frequencies / sums[:,None]
        
        return feature_frequencies
