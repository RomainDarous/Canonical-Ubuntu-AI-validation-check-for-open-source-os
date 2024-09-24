import nltk
import json
from pathlib import Path
import sys
import os
import pandas as pd

class Processor :

    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"
    VALID_LANGUAGES = "languages"
    DATASET_FOLDER = Path('../1_data_collection/os_by_language/dataset')

    def __init__(self, metadata_path) :
        try :
            with open(metadata_path, 'r', encoding='utf-8') as f :
                self.metadata = json.load(f)
        except Exception as e :
            print("Error while loading the metadata file. Please try again.")
            sys.exit()

    # ------------------ MAIN FUNCTIONS ----------------------------- #
    def data_cleaning(self, update_only = False) :

        # Checking if the cleaning mut be performed on a subset of the dataset
        if update_only : 
            list_dir = self.metadata[self.UPDATED_FILES] # dict
            
        else : 
            top_list_dir = os.listdir(self.DATASET_FOLDER)
            list_dir = [os.listdir(folder) for folder in top_list_dir if os.path.isdir(folder)] # list

        # Start of the cleaning process
        for file in list_dir :
            df = pd.read_csv(file, encoding='utf-8')

            df.dropna(subset=["en"], inplace=True)

            if update_only : working_df = df[list_dir[file]]
            else : working_df = df

            # Removing empty strings :
            working_df

        pass

    def data_fusion(self) :
        pass

    def data_resampling(self) :
        pass

    def data_splitting(self) :
        pass
    
    # ----------------- HELP FUNCTIONS ------------------------------- #
    def load_file_update(self, metadata_path) :

    
    