import json
from pathlib import Path
import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
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
            list_dir = [] # list
            for sub_folder in top_list_dir :
                sub_files = os.listdir(self.DATASET_FOLDER / sub_folder)
                list_dir.extend([self.DATASET_FOLDER / sub_folder / sub_file for sub_file in sub_files])

        # Start of the cleaning
        for file in list_dir :
            df = pd.DataFrame()
            for delimiter in ['|',','] :
                try : df = pd.read_csv(file, dtype=str, encoding='utf-8', delimiter=delimiter)
                except : continue

            if df.empty :
                print(f"Impossible to open {file}")
                continue

            # Removing empty strings
            df.replace('', np.nan, inplace=True)
            df.dropna(subset=["en"], inplace=True)
            
            """# Checking that the file is not empty (it shouldn't if data collectin is correctly performed)
            if df.iloc[1:].empty :
                os.remove(file)
                continue"""

            if update_only : working_df = df[list_dir[file]]
            else : working_df = df


            # Cleaning function
            cleaned_working_df = self.cleaning(working_df)

            # Final replacement
            if update_only : df[list_dir[file]] = cleaned_working_df
            else : df = cleaned_working_df

            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)
            
            # Saving the cleaned file
            if df.iloc[1:].empty : 
                os.remove(file)
                continue
            else : df.to_csv(file, encoding='utf-8', index = False, sep='|')

        return

    def data_fusion(self) :
        # REMOVE THE DATE et les guillements
        pass

    def data_resampling(self) :
        pass

    def data_splitting(self) :
        pass
    
    # ----------------- HELP FUNCTIONS ------------------------------- #
    def cleaning(self, df) -> pd.DataFrame:

        # Removing identical pairs
        for column in df.columns :
            if column != 'en' :
                df[column] = df[column][df[column] != df['en']]

        for column in df.columns :
            # Removing indentical pairs
            if column != 'en' :
                df = df[df[column] != df['en']]
                df = df[(df[column] != column) & (df['en'] != 'en')]

            # Return line
            df[column] = df[column].str.replace(r'(\n|\r|\t)(\n|\r|\t)*', ' ', regex=True)
            df[column] = df[column].str.replace(r'(\\n|\\r|\\t)(\\n|\\r|\\t)*', ' ', regex=True)

            # HTML boxes and links
            df[column] = df[column].apply(self.remove_html)
            df[column] = df[column].str.replace(r'http\S+|www\S+', '', regex=True)
            
            # Special characters and UTF wrong characters
            df[column] = df[column].str.replace(r'\[UTF-[^\]]+\]', '', regex=True)

            # Additional removals
            df[column] = df[column].str.replace(r'\s*@\w+', ' ', regex=True)
            df[column] = df[column].str.replace(r'(\()*(%[^)]+)(\))*', '', regex=True) # (%s) characters
            df[column] = df[column].str.replace(r'(\$)*\{[^}]+\}', ' ', regex=True) # ${} characters
            df[column] = df[column].str.replace(r'\s*[\(\[]\s*[\)\]]\s*', '', regex=True) # {}, () empty, etc.
            df[column] = df[column].str.replace(r'(\[)*|(\])*', '', regex=True) # all the [ and ] brackets

            df[column] = df[column].str.replace(r'(?<=\s)([^\w\s]+)(?=\s)|^([^\w\s]+)(?=\s)|(?<=\s)([^\w\s]+)$', ' ', regex=True) # sequence of special characters removed
            df[column] = df[column].str.replace(r'(`)*|(\')*|(")*|(”)*|(“)*', '', regex=True) # Quotation marks
            df[column] = df[column].str.replace(r'^-+|-', ' ', regex=True) # --output-only -> output only

            # Removing excessive spaces
            df[column] = df[column].str.replace(r'\s+', ' ', regex=True)
            df[column] = df[column].str.strip()
    
        return df
    
    def remove_html(self, sentence: str) :
        if type(sentence) == str : return BeautifulSoup(sentence, 'html.parser').get_text()
        else : return sentence
    


    
    