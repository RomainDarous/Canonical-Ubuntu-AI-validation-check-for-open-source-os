import json
from pathlib import Path
import re
import sys
import os
import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import nltk
from nltk.corpus import wordnet
from typing import Dict
from datasets import Dataset, DatasetDict, concatenate_datasets
from sentence_transformers import SentenceTransformer

import torch


class Processor:
    """A class used to collect translations from Weblate and Ubuntu Launchpad.

    Attributes:
        METADATA_FILE_01 (Path): Path to the metadata file of the collected raw os translations.
        UPDATED_FILES (str): Key for updated files in the metadata file of the collected raw os translations.
        ARCH_VERSIONS (str): Key for archive versions in the metadata file of the collected raw os translations.
        VALID_LANGUAGES (str): Key for the list of accepted languages for translations in the metadata file of the collected raw os translations.
        
        METADATA_FILE_02 (Path): Path to the metadata file of the processed os translations.
        COLLECTED_DATASET_FOLDER (Path): Path to the collected raw os translations
        MERGED_DATASET (Path): Path to the processed os translations dataset.
        LAST_MERGED_FILE (Path): Path to the last merged file to resume merging.
        LAST_CLEANED_FILE (Path): Path to the last cleaned file to resume cleaning.
    """


    # Collected data metadata
    METADATA_FILE_01 = Path('../1_data_collection/os_by_language/metadata_01.json')
    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"
    VALID_LANGUAGES = "languages"
    #COLLECTED_DATASET_FOLDER = Path('../1_data_collection/os_by_language/dataset')
    #COLLECTED_DATASET_FOLDER = Path('./tmp/dataset/')
    
    MERGED_DATASET = Path('./2_os_by_language/datasets')
    METADATA_FILE_02 = Path('./2_os_by_language/metadata_02.json')
    ROW_NUMBER = "raws_per_file"
    LAST_MERGED_FILE = Path('./2_os_by_language/last_merged_file.txt')
    LAST_CLEANED_FILE = Path('./2_os_by_language/last_cleaned_file.txt')

    COLLECTED_DATASET_FOLDER = MERGED_DATASET
    #MERGED_DATASET = Path('./tmp/dataset/')

    # Translation check-up
    TRANSLATION_CHECK_FILE = Path('./2_os_by_language/02_translation_check.json')
    LAST_CHECKED_FILE = 'last_checked_file'

    def __init__(self) -> None:
        """Initializes the Processor instance and loads the metadata of the collected raw os translation.
        """
        # Downloading nltk databases
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        
        self.list_dir = None

        # Loading the metadata files
        try:
            with open(self.METADATA_FILE_01, 'r', encoding='utf-8') as f:
                self.metadata_01 = json.load(f)
            with open(self.METADATA_FILE_02, 'r', encoding='utf-8') as f:
                self.metadata_02 = json.load(f)
            with open(self.TRANSLATION_CHECK_FILE, 'r', encoding='utf-8') as f :
                self.translation_check_metadata = json.load(f)
        except Exception as e:
            print("Error while loading the metadata file. Please try again.")
            sys.exit()

    def data_cleaning(self, update_only: bool =False) -> None:
        """Performs cleaning of the dataset.

        Args:
            update_only (bool, optional): True if you want to clean only updated files from the dataset. Defaults to False.
        """

        print("STARTING THE CLEANING PROCESS...")

        # Checking if the cleaning must be performed on a subset of the dataset
        if update_only: 
            self.list_dir = [f'..\\1_data_collection\\{file}' for file in self.metadata_01[self.UPDATED_FILES]]  # dict
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

                if df.empty and i%20 == 0:
                    print(f"Empty file: {file}")
                    continue
 
                # Removing empty strings
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)

                # Cleaning function
                working_df = df.loc[:, :]
                cleaned_working_df = self.cleaning(working_df)

                # Final replacement
                df = cleaned_working_df.loc[:, :]
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)

                # Stack small rows together
                df = self.row_stacking(df).loc[:, :]

                if update_only: del self.metadata_01[self.UPDATED_FILES][i]

                # Saving the cleaned file
                if df.empty:
                    os.remove(file)
                    continue
                else: 
                    df['score'] = 1 # adding a label column
                    df.to_csv(file, encoding='utf-8', index=False, sep='\t')

            # Deleting the resume file
            if os.path.exists(self.LAST_CLEANED_FILE) : os.remove(self.LAST_CLEANED_FILE)

        except Exception as e :
            print(f"DATA CLEANING exception : {e}")
            with open(self.LAST_CLEANED_FILE, "w", encoding='utf-8') as f :
                f.write(str(file))
            
        finally :
            with open(self.METADATA_FILE_01, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_01, f, indent=4)
            return

    
    def data_merging(self) -> None :
        """ Merges all the files to build monolingual datasets

        """
        print("STARTING THE MERGING PROCESS...")
        # Start of the fusion
        dataset_dict : Dict[str, pd.DataFrame] = {}

        # Checking if the cleaning must be performed on a subset of the dataset
        self.list_dir = self.load_collected_files()
        
        if len(self.list_dir) == 0 : 
            print("No datafile found to merge")
            return
        
        file = self.list_dir[0]

        try :
            last_idx = self.resume(self.LAST_MERGED_FILE)
            
            # Concatenating files of the same language
            for i in range(last_idx, len(self.list_dir)):
                file = self.list_dir[i]
                if not os.path.exists(file) : 
                    print("File not found : {file}")
                    continue
                if i%50 == 0 : print(f'File being merged : {file}')

                code = str(file.with_suffix('')).split('-')[-1]
                if code not in dataset_dict : dataset_dict[code] = pd.DataFrame()

                # Opening the file to merge and merging
                tmp_df = pd.DataFrame()
                for delimiter in ['|', ',', '\t']:
                    try: 
                        tmp_df = pd.read_csv(file, dtype=str, encoding='utf-8', delimiter=delimiter)
                        assert(len(tmp_df.columns) > 1)
                        break
                    except: continue

                if tmp_df.empty:
                    print(f"Impossible to open / Empty : {file}")
                    continue
                else : dataset_dict[code] = pd.concat([dataset_dict[code], tmp_df], ignore_index=True)

            # Deleting the resume file
            if os.path.exists(self.LAST_MERGED_FILE) : os.remove(self.LAST_MERGED_FILE)
        except Exception as e :
            print(f"DATA MERGING ERROR : {e}")
            with open(self.LAST_MERGED_FILE, "w", encoding='utf-8') as f :
                f.write(str(file))
        
        finally :
            # Saving merged datasets
            self.MERGED_DATASET.mkdir(parents=True, exist_ok=True)
            for code in dataset_dict :
                dataset_dict[code].drop_duplicates(inplace=True)
                if not dataset_dict[code].empty :
                    dataset_dict[code]['lang'] = code # adding a language rwo for future full merging
                    dataset_dict[code].rename(columns={dataset_dict[code].columns[0]: 'sentence1', dataset_dict[code].columns[1]: 'sentence2'}, inplace=True)
                    dataset_dict[code].to_csv(self.MERGED_DATASET / f'os-dataset-{code}.csv', index=False, sep='\t')
                    self.metadata_02[self.ROW_NUMBER][str(self.MERGED_DATASET / f'os-dataset-{code}.csv')] = dataset_dict[code].shape[0]
            
            # Saving the metadatas in the file
            with open(self.METADATA_FILE_02, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_02, f, indent=4)
            return
        
    def preliminary_translation_quality_check(self) -> None :
        print("PRELIMINARY TRANSLATION QUALITY CHECK...")

        model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
        self.list_dir = os.listdir(self.MERGED_DATASET)

        threshold = 0.1
        resumed = True

        """if not self.translation_check_metadata[self.LAST_CHECKED_FILE] : self.reset_translation_metadata()
        else : resumed = False"""
        resumed = False # TO CHANGE

        try :
            # Checking in all files low similarity translations
            for file in self.list_dir :
                self.translation_check_metadata[self.LAST_CHECKED_FILE] = file

                path = self.MERGED_DATASET / Path(file)
                print("Current path : ", path)
                if not resumed and path != self.MERGED_DATASET / Path(self.translation_check_metadata[self.LAST_CHECKED_FILE]) : continue
                else : resumed=True
                df = pd.read_csv(path, delimiter='\t')
                sentence1_embedding = model.encode(list(df['sentence1']))
                sentence2_embedding = model.encode(list(df['sentence2']))

                for i in range(len(sentence1_embedding)) :
                    cos_sim = self.cosim(sentence1_embedding[i], sentence2_embedding[i])
                    if cos_sim < threshold :
                        self.translation_check_metadata[file].append(i)
                
                with open(self.TRANSLATION_CHECK_FILE, 'w', encoding='utf-8') as f :
                    json.dump(self.translation_check_metadata, f, indent=4)
                    print("02_translation_check.json updated !")
                    print(f"Number of mismatches found : {len(self.translation_check_metadata[file])}")
                
                
            # Resetting last checked file to default
            self.translation_check_metadata[self.LAST_CHECKED_FILE] = ''

        except Exception as e :
            print(f"An error occured : {e}")

        finally :
            # Saving the potential mistranslated strings in a file to manually check
            with open(self.TRANSLATION_CHECK_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.translation_check_metadata, f, indent=4)

        return
    
    def delete_wrong_translations(self) -> None :
        for file in self.translation_check_metadata.keys() :
            if file != self.LAST_CHECKED_FILE :
                df = pd.read_csv(self.MERGED_DATASET / file, encoding='utf-8', delimiter='\t')
                indices_to_delete = self.translation_check_metadata[file]
                df.drop(indices_to_delete, inplace=True)
                df.to_csv(self.MERGED_DATASET / file, encoding='utf-8', index=False, sep='\t')

                # Update its line number
                self.metadata_02[self.ROW_NUMBER][str(self.MERGED_DATASET / file)] = df.shape[0]
        
        with open(self.METADATA_FILE_02, 'w', encoding=('utf-8')) as f :
            json.dump(self.metadata_02, f, indent=4)
        return

    def data_upload(self, data_dir) -> None :
        """Once the processing is done, pushes the dataset to HuggingFace
        """
        files = os.listdir(data_dir)

        datasets = []

        # Load each file and add it as a subset
        for filename in files:
            file_path = os.path.join(data_dir, filename)
            if filename.endswith(".csv"):  # Modify if using another file format
                df = pd.read_csv(file_path, delimiter='\t')  # Load file into a DataFrame
                subset = Dataset.from_pandas(df)  # Convert DataFrame to Dataset
                datasets.append(subset)
        
        dataset = concatenate_datasets(datasets)

        # Define repository name and upload the dataset
        repo_name = input("Dataset directory: ")
        token=input("Your personal token: ")

        # Push the dataset to the Hugging Face Hub
        dataset.push_to_hub(repo_name, token=token)



    # ----------------- HELP FUNCTIONS ------------------------------- #
    def cosim(self, vec1, vec2) -> float :
        vec1 = torch.tensor(vec1)
        vec2 = torch.tensor(vec2)
        dot_product = torch.dot(vec1, vec2)  # Efficient dot product
        norm_vec1 = torch.linalg.norm(vec1)  # Norm of vec1
        norm_vec2 = torch.linalg.norm(vec2)  # Norm of vec2
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity.item()

    def row_stacking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gathers small rows together (less than three words)

        Args:
            df (pd.DataFrame): The dataframe to consider

        Returns:
            pd.DataFrame: The factorized dataset. Rows are alignes, separated by '. ' for each feature
        """
        l = len(df)

        source_feature = []
        target_feature = []
        row_nb = 0

        code1, code2 = df.columns[:2]

        for index, row in df.iterrows():
            if len(row[code1].split(' ')) <= 4 and '?' not in row[code1] and '!' not in row[code1] :
                df = df.drop(index=index).loc[:, :]
                l -= 1
                source_feature.append(row[code1].replace('.',''))
                target_feature.append(row[code2].replace('.',''))
                row_nb += 1

            if row_nb == 5 :
                df.loc[l, [code1, code2]] = [('. '.join(source_feature)), ('. '.join(target_feature))]
                source_feature = []
                target_feature = []
                l += 1
                row_nb = 0

        return df

    def cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Performs cleaning of the dataset in input for training.

        Args:
            df (pd.Dataframe): the input dataframe to clean
        Returns:
            pd.DataFrame: a cleaned version of the input dataframe
        """
        type_clean = [
            r'@\w+\s'                               # Remove @"content "
            r'\n(\n)*|\t(\t)*|\\n(\\n)*',
            r'http\S+|www\S+',                      # Remove URLs
            r'\[UTF-[^\]]+\]',                      # Remove UTF characters
            r'(\()*(%[^)]+)(\))*',                  # Remove (%s) characters
            r'^-+|-',                               # Replace leading/trailing dashes
            r'(?<!^)(?=[A-Z])(?![A-Z])',            # Split attached words
            r'\b(?:\d{1,3}\.){3}\d{1,3}\b',         # IPv4
            r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b', # IPv6
            r'\b(?:\d+\.)+\d+\b',                    # Removing 4.6.5.3-like sequences
        ]

        char_clean = [
            r'(\$)',                                    # Remove ($){} characters
            r'\(\((\()*((\s)*)(\))* | (\()*((\s)*)(\))*\)\) | \s\(\s | \s\)\s | \((\()*((\s)*)(\))*\)',   # Remove ((())))) types of strings
            r'\{\{(\{)*((\s)*)(\})* | (\{)*((\s)*)(\})*\}\} | \s\{\s | \s\}\s | \{(\{)*((\s)*)(\})*\}',   # Same for {
            r'\[\[(\[)*((\s)*)(\])* | (\[)*((\s)*)(\])*\]\] | \s\[\s | \s\]\s | \[(\[)*((\s)*)(\])*\]',   # Same for {
            r'\/(\/)*',                                 # Remove all /(/)*
            r'(?<=\s)([^\w\s?!:;]+)(?=\s)|^([^\w\s?!:;]+)(?=\s)|(?<=\s)([^\w\s?!:;]+)$',  # Remove special characters
            r'``*| \'(\')*|""*|””*|““*',                # Remove various quotation marks
        ]

        for column in df.columns:
            # Removing identical pairs
            if 'en' in df.columns :
                if column != 'en':
                    df = df[(df.loc[:,column] != df['en']) & (df.loc[:,column] != column) & (df['en'] != 'en')]

                else :
                    # Checking that the English version contains proper English words
                    df.loc[:,column] = df.loc[:,column].apply(self.is_english)
            
            # Cleaning all the columns
            try : 
                df.loc[:,column] = df.loc[:,column].apply(self.remove_html)
                df.loc[:,column] = df.loc[:,column].str.replace('|'.join(type_clean), ' ', regex=True)
                df.loc[:,column] = df.loc[:,column].str.replace('|'.join(char_clean), ' ', regex=True)

                # Removing excessive spaces
                df.loc[:,column] = df.loc[:,column].str.replace(r'\t(\t)*', ' ', regex=True)
                df.loc[:,column] = df.loc[:,column].str.replace(r'\s+', ' ', regex=True).str.strip()
            except Exception as e :
                print(f"Error on column {column} : {e}")
                print(df.head())
                continue
        
        # Removing duplicates
        df = (df.drop_duplicates(inplace=False)).loc[:, :]

        if 'en' in df.columns :
            return df[df['en'].str.split(' ').str.len() <= 128]
        else :
            return df
    
    def remove_html(self, sentence: str) -> str:
        """Removes html content from the data.

        Args:
            sentence (str): The string to process.

        Returns:
            str: The string free from any html code.
        """
        if isinstance(sentence, str): 
            return BeautifulSoup(sentence, 'html.parser').get_text()
        return sentence
    

    def is_english(self, sentence: str) -> str:
        """Checks that the english string is not only made of acronyms that have no translation

        Args:
            sentence (str): the string to check

        Returns:
            str: the original string if more than a third of words are english, an empty string otherwise
        """
        if not sentence : return ''
        words = nltk.word_tokenize(sentence)
        en_count = np.sum([1 if bool(wordnet.synsets(word.lower())) else 0 for word in words])
        
        if en_count / len(words) > 0.33: 
            return sentence
        else : return ''
    
    def resume(self, resume_file: Path) -> int :
        """Resume data concatenation or merging if interrupted

        Args:
            resume_file (Path): the last file that has to be checked

        Returns:
            int: the idx on the file list to process corresponding to the file
        """
        try :
            last_file = None
            if os.path.exists(resume_file) :
                with open(resume_file, 'r', encoding='utf-8')as f :
                    last_file = f.readline()
            else : return 0

            last_idx = 0
            if last_file and self.list_dir:
                while self.list_dir[last_idx] != last_file : last_idx += 1
            return last_idx
        except : return 0

    def load_collected_files(self) -> list :
        """Gets all file paths of the dataset into a list

        Returns:
            list: The list of the files
        """
        top_list_dir = os.listdir(self.COLLECTED_DATASET_FOLDER)
        list_dir = []  # list
        for sub_folder in top_list_dir:
            if not (self.COLLECTED_DATASET_FOLDER / sub_folder).is_dir(): list_dir.append(self.COLLECTED_DATASET_FOLDER / sub_folder)
            else :
                sub_files = os.listdir(self.COLLECTED_DATASET_FOLDER / sub_folder)
                list_dir.extend([self.COLLECTED_DATASET_FOLDER / sub_folder / sub_file for sub_file in sub_files])
        return list_dir
    

    #---------------------------- METADA MANAGEMENT ---------------------------------#
    def empty_updated_files(self) -> None :
        """Empties the list of updated files in the metadata and saves it to the metadata file."""

        self.metadata_01[self.UPDATED_FILES] = []
        with open(self.METADATA_FILE_01, "w", encoding='utf-8') as f :
            json.dump(self.metadata_01, f, indent=4)

    def reset_translation_metadata(self) -> None :
        self.list_dir = os.listdir(self.MERGED_DATASET)
        for key in self.list_dir :
            self.translation_check_metadata[key] = []
        self.translation_check_metadata[self.LAST_CHECKED_FILE] = ''

        with open(self.TRANSLATION_CHECK_FILE, 'w', encoding='utf-8') as f :
            json.dump(self.translation_check_metadata, f, indent=4)