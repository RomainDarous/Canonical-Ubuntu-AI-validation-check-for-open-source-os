import json
from pathlib import Path
import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import nltk
from nltk.corpus import wordnet
from typing import Dict

class Processor:
    # Collected data metadata
    METADATA_FILE_COL = Path('../1_data_collection/os_by_language/dataset_metadata.json')
    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"
    VALID_LANGUAGES = "languages"
    #COLLECTED_DATASET_FOLDER = Path('../1_data_collection/os_by_language/dataset')
    COLLECTED_DATASET_FOLDER = Path('./tmp/dataset/')

    
    MERGED_DATASET = Path('./2_os_by_language/datasets')
    METADATA_FILE_PROC = Path('./2_os_by_language/2_dataset_metadata.json')
    LAST_MERGED_FILE = Path('./2_os_by_language/last_merged_file.txt')
    LAST_CLEANED_FILE = Path('./2_os_by_language/last_cleaned_file.txt')

    def __init__(self):
        # Downloading nltk databases
        nltk.download('punkt')
        nltk.download('wordnet')

        self.list_dir = None

        try:
            with open(self.METADATA_FILE_COL, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print("Error while loading the metadata file. Please try again.")
            sys.exit()

    def data_cleaning(self, update_only=False):
        print("STARTING THE CLEANING PROCESS...")
        # Checking if the cleaning must be performed on a subset of the dataset
        if update_only: 
            self.list_dir = self.metadata[self.UPDATED_FILES]  # dict
        else: 
            self.list_dir = self.load_collected_files()

        if len(self.list_dir) == 0 : 
            print("No datafile found to merge")
            return
        
        file = self.list_dir[0]

        last_idx = self.resume(self.LAST_CLEANED_FILE)

        try :
            # Start of the cleaning
            for i in range(last_idx, len(self.list_dir)):
                file = self.list_dir[i]
                if i%20 == 0 : print(f'File being cleaned : {file}')

                df = pd.DataFrame()
                for delimiter in ['|', ',', '\t']:
                    try: 
                        df = pd.read_csv(file, dtype=str, encoding='utf-8', delimiter=delimiter)
                        assert(len(df.columns) > 1)
                        break
                    except: continue

                if df.empty:
                    print(f"Impossible to open {file}")
                    continue

                # Removing empty strings
                df.replace('', np.nan, inplace=True)
                df.dropna(subset=["en"], inplace=True)

                if update_only: 
                    working_df = df.loc[:, self.list_dir[file]]
                else: 
                    working_df = df

                # Cleaning function
                cleaned_working_df = self.cleaning(working_df)

                # Final replacement
                if update_only: 
                    df.loc[:, self.list_dir[file]] = cleaned_working_df
                else: 
                    df = cleaned_working_df

                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)
                
                # Saving the cleaned file
                if df.empty: 
                    os.remove(file)
                    continue
                else: 
                    df.to_csv(file, encoding='utf-8', index=False, sep='\t')

            # Deleting the resume file
            if os.path.exists(self.LAST_CLEANED_FILE) : os.remove(self.LAST_CLEANED_FILE)
        except Exception as e :
            print(f"Exception {e} occured")
            with open(self.LAST_CLEANED_FILE, "w", encoding='utf-8') as f :
                f.write(str(file))
            
        finally :
            return

    
    def data_merging(self) :
        print("STARTING THE MERGING PROCESS...")
        # Start of the fusion
        dataset_dict : Dict[str, pd.DataFrame] = {}

        # Checking if the cleaning must be performed on a subset of the dataset
        if not self.list_dir: self.list_dir = self.load_collected_files()
        
        if len(self.list_dir) == 0 : 
            print("No datafile found to merge")
            return
        
        file = self.list_dir[0]

        try :
            last_idx = self.resume(self.LAST_MERGED_FILE)
            
            # Concatenating files of the same language
            for i in range(last_idx, len(self.list_dir)):
                file = self.list_dir[i]
                if i%20 == 0 : print(f'File being merged : {file}')

                code = str(file.with_suffix('')).split('-')[-1]
                if code not in dataset_dict : dataset_dict[code] = pd.DataFrame()
                print(file)
                tmp_df = pd.read_csv(file, delimiter='\t')
                dataset_dict[code] = pd.concat([dataset_dict[code], tmp_df], ignore_index=True)

            # Deleting the resume file
            if os.path.exists(self.LAST_MERGED_FILE) : os.remove(self.LAST_MERGED_FILE)
        except Exception as e :
            print(f"The error {e} occured")
            with open(self.LAST_MERGED_FILE, "w", encoding='utf-8') as f :
                f.write(str(file))
        
        finally :
            # Saving merged datasets
            for code in dataset_dict :
                dataset_dict[code].to_csv(self.MERGED_DATASET / f'os-dataset-{code}.csv', index=False)
            return




    # ----------------- HELP FUNCTIONS ------------------------------- #
    def cleaning(self, df) -> pd.DataFrame:
        type_clean = [
            r'http\S+|www\S+',                      # Remove URLs
            r'\[UTF-[^\]]+\]',                      # Remove UTF characters
            r'\s*@\w+',                             # Remove mentions like @username
            r'(\()*(%[^)]+)(\))*',                  # Remove (%s) characters
            r'^-+|-',                               # Replace leading/trailing dashes
            r'(?<!^)(?=[A-Z])(?![A-Z])',            # Split attached words
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',         # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b', # IPv6
            r'\b(?:\d+\.)+\d+\b',                    # Removing 4.6.5.3 like sequences
        ]

        char_clean = [
            r'(\$)*\{[^}]+\}',                      # Remove ($){} characters
            r'\((\()*\)(\))*',                      # Remove ((())))) types of strings
            r'\[(\[)*|\](\])*|\/(\/)*|\(\)*',      # Remove all [ and ] brackets
            r'(?<=\s)([^\w\s?!]+)(?=\s)|^([^\w\s?!]+)(?=\s)|(?<=\s)([^\w\s?!]+)$',  # Remove special characters
            r'``*| \'(\')*|""*|””*|““*',           # Remove various quotation marks
        ]

        for column in df.columns:
            # Removing identical pairs
            if column != 'en':
                df = df[(df.loc[:,column] != df['en']) & (df.loc[:,column] != column) & (df['en'] != 'en')]

            df.loc[:,column] = df.loc[:,column].apply(self.remove_html)
            df.loc[:,column] = df.loc[:,column].str.replace('|'.join(type_clean), ' ', regex=True)
            df.loc[:,column] = df.loc[:,column].str.replace('|'.join(char_clean), ' ', regex=True)

            # Removing excessive spaces
            df.loc[:,column] = df.loc[:,column].str.replace(r'\s+', ' ', regex=True).str.strip()

            # Checking that the English version contains proper English words
            df.loc[:,column] = df.loc[:,column].apply(self.is_english)

        return df
    
    def remove_html(self, sentence: str) -> str:
        if isinstance(sentence, str): 
            return BeautifulSoup(sentence, 'html.parser').get_text()
        return sentence

    def is_english(self, sentence: str) -> str:
        words = nltk.word_tokenize(sentence)
        en_count = np.sum([1 if bool(wordnet.synsets(word.lower())) else 0 for word in words])
        if en_count / len(words) > 0.33: 
            return sentence
        return ''
    
    def resume(self, resume_file: Path) -> int :
        try :
            # Resuming data concatenation if interrupted
            last_file = None
            if os.path.exists(resume_file) :
                with open(resume_file, 'r', encoding='utf-8')as f :
                    last_file = f.readline()

            last_idx = 0
            if last_file and self.list_dir:
                while self.list_dir[last_idx] != last_file : last_idx += 1
            return last_idx
        except : return 0

    def load_collected_files(self) :
        top_list_dir = os.listdir(self.COLLECTED_DATASET_FOLDER)
        list_dir = []  # list
        for sub_folder in top_list_dir:
            sub_files = os.listdir(self.COLLECTED_DATASET_FOLDER / sub_folder)
            list_dir.extend([self.COLLECTED_DATASET_FOLDER / sub_folder / sub_file for sub_file in sub_files])
        return list_dir
