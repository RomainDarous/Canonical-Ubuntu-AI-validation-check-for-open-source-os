import nltk
import json
from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np

class Processor :

    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"
    VALID_LANGUAGES = "languages"
    #DATASET_FOLDER = Path('../1_data_collection/os_by_language/dataset')
    DATASET_FOLDER = Path('./tmp/dataset')
    METADATA_FILE = Path('../1_data_collection/os_by_language/dataset_metadata.json')


    def __init__(self) :
        try :
            with open(self.METADATA_FILE, 'r', encoding='utf-8') as f :
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
            print(top_list_dir)
            list_dir = [] # list
            for sub_folder in top_list_dir :
                sub_files = os.listdir(self.DATASET_FOLDER / sub_folder)
                list_dir.extend([self.DATASET_FOLDER / sub_folder / sub_file for sub_file in sub_files])
            print("passed", list_dir)

        # Start of the cleaning
        for file in list_dir :
            print(file)
            df = pd.read_csv(file, encoding='utf-8')

            # Removing empty strings
            df.replace("", np.nan, inplace=True)
            df.dropna(inplace=True)
            df = df.reset_index(drop=True)
            
            if df.iloc[1:].empty :
                os.remove(file)
                continue

            if update_only : working_df = df[list_dir[file]]
            else : working_df = df

            # Cleaning function
            cleaned_working_df = self.cleaning(working_df)

            # Final replacement
            if update_only : df[list_dir[file]] = cleaned_working_df
            else : df = cleaned_working_df

            # Saving the cleaned file
            df.to_csv(file, encoding='utf-8')

        return

    def data_fusion(self) :
        # REMOVE THE DATE
        pass

    def data_resampling(self) :
        pass

    def data_splitting(self) :
        pass
    
    # ----------------- HELP FUNCTIONS ------------------------------- #
    def cleaning(self, df) -> pd.DataFrame:

        # Initial refactoring modifications
        df.replace("\n", " ", inplace=True)
        return df.reset_index(drop=True)


    
    