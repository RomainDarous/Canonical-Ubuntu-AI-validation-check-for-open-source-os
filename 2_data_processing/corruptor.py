from pathlib import Path
from typing import Dict
from datasets import DatasetDict, load_dataset, Dataset, concatenate_datasets
from pandas import DataFrame
import torch
import json
import sys
import os
import pandas as pd
import numpy as np
import re
from processor import Processor
import traceback


class Corruptor:

    MERGED_DATASET = Path('./2_os_by_language/datasets')
    METADATA_FILE_02 = Path('./2_os_by_language/metadata_02.json')
    CORRUPTED_FILES = "corrupted_files"
    ROW_NUMBER = "raws_per_file"
    #MERGED_DATASET = Path('./2_os_by_language/datasets')
    MERGED_DATASET = Path('./tmp/dataset')

    HATEFUL_DATASET = Path('./multilingual_hateful_sets/data')
    PROCESSED_HATEFUL_DATASET = Path('./multilingual_hateful_sets/processed_data')
    HATE_COLUMN_NAME = "sentence"
    HATESET_METADATA = Path('./multilingual_hateful_sets/hatespeech_metadata.json')
    CLOSE_LANGUAGES = "closest_languages"
    HATE_LAST_ID = "last_checked_id_per_hate_set"

    ACCEPTED_LANGUAGES = [
        "Arabic", "Basque", "Breton", "Catalan", "Chinese_China", "Chinese_Hongkong", 
        "Chinese_Taiwan", "Chuvash", "Czech", "Dhivehi", "Dutch", "English", 
        "Esperanto", "Estonian", "French", "Frisian", "Georgian", "German", "Greek", 
        "Hakha_Chin", "Indonesian", "Interlingua", "Italian", "Japanese", "Kabyle", 
        "Kinyarwanda", "Kyrgyz", "Latvian", "Maltese", "Mongolian", "Persian", "Polish", 
        "Portuguese", "Romanian", "Romansh_Sursilvan", "Russian", "Sakha", "Slovenian", 
        "Spanish", "Swedish", "Tamil", "Tatar", "Turkish", "Ukranian", "Welsh"
    ]

        
    def __init__(self) -> None:
        # Loading the metadata files
        try:
            with open(self.METADATA_FILE_02, 'r', encoding='utf-8') as f:
                self.metadata_02 = json.load(f)
            with open(self.HATESET_METADATA, 'r', encoding='utf-8') as f :
                self.hateset_metadata = json.load(f)
        except Exception as e:
            print(f"Error while loading the metadata file : {self.METADATA_FILE_02}.\n Please try again.")
            sys.exit()

        # Loading hatespeech datasets
        self.mult_hate_speech = DatasetDict()

    def data_corruption(self) -> None :
        print("START OF THE CORRUPTION PROCESS...")
        self.list_dir = os.listdir(self.MERGED_DATASET)

        # Getting the max number of rows
        max_row_number = max(self.metadata_02[self.ROW_NUMBER].values())
        try : 
            for file in self.list_dir :
                try :
                    language = (file.split('-')[-1]).split('.')[0]

                    ### Taking hate sets with close language structures
                    hate_languages = self.hateset_metadata[self.CLOSE_LANGUAGES][language]
                    df = self.corrupt(self.MERGED_DATASET / file, hate_languages, max_row_number)
                    df.to_csv(self.MERGED_DATASET / file, encoding='utf-8', index=False, sep='\t')

                    self.metadata_02[self.CORRUPTED_FILES].append(file)
                except Exception as e :
                    print(f"Error with file {file} : {e}")
                    traceback.print_exc()
            print("DATA SUCCESSFULLY CORRUPTED !")

        except Exception as e : 
            print(f"Error code : {e}")

        finally :
            with open(self.HATESET_METADATA, 'w', encoding='utf-8') as f:
                json.dump(self.hateset_metadata, f, indent=4)

            with open(self.METADATA_FILE_02, 'w', encoding='utf-8') as f :
                json.dump(self.metadata_02, f, indent=4)
        return

    def corrupt(self, file: Path, hate_languages: list[str], max_row_number: int) -> DataFrame :
        print(f"Corrupting : {file}...")
        # Loading the dataset
        df = pd.read_csv(file, encoding='utf-8', delimiter='\t')
        df = df.sample(frac=1, random_state=np.random.randint(5_000)).reset_index(drop=True)

        # Dataset features
        nb_0_labels = np.sum(df['score'] == 0)
        nb_1_labels = np.sum(df['score'] == 1)
        init_os_set_len = df.shape[0]
        os_set_len = df.shape[0]
        curr_os_set_idx = 0


        # Loading the hate set
        curr_hate_language_idx = 0
        hate_df = pd.read_csv(self.PROCESSED_HATEFUL_DATASET / f"hatefulspeech_{hate_languages[0]}.csv", sep='\t', encoding='utf-8')
        
        if hate_languages[0] in self.hateset_metadata[self.HATE_LAST_ID] : curr_hate_set_idx = self.hateset_metadata[self.HATE_LAST_ID][hate_languages[0]]
        else : curr_hate_set_idx = 0
        len_hate_set = len(hate_df)


        # The potential additional corrupted rows to add
        new_rows = {'sentence1' : [],
                    'sentence2' : [],
                    'score' : [],
                    'sentence2_language' : []}


        steps = 0
        init_idx_corrupted = []
        while nb_0_labels < int(2*nb_1_labels/3) :
            # Switching to a new hate set if required
            if steps == len_hate_set :
                curr_hate_language_idx += 1
                if curr_hate_language_idx > len(hate_languages) : break
                hate_df = pd.read_csv(self.PROCESSED_HATEFUL_DATASET / f"hatefulspeech_{hate_languages[curr_hate_language_idx]}.csv", sep='\t', encoding='utf-8')
                if hate_languages[0] in self.hateset_metadata[self.HATE_LAST_ID] : curr_hate_set_idx = self.hateset_metadata[self.HATE_LAST_ID][hate_languages[0]]
                else : curr_hate_set_idx = 0
                len_hate_set = len(hate_df)
                steps = 0


            # Incorporate hatespeech translations
            sentence1 = str(df.loc[curr_os_set_idx, 'sentence1'])
            sentence2 = str(df.loc[curr_os_set_idx, 'sentence2'])
            corrupted_sentence2 = sentence2

            hate_speech = hate_df['sentence'][curr_hate_set_idx]

            hate_speech_list = hate_speech.split(' ')
            sentence2_list = sentence2.split(' ')

            if len(hate_speech_list) >= len(sentence2_list) :
                corrupted_sentence2 = hate_speech
            else :
                start_idx = np.random.randint(0, len(sentence2_list) - len(hate_speech_list))
                corrupted_sentence2 = ' '.join(sentence2_list[:start_idx]) + ' ' + hate_speech.lower() + ' ' + ' '.join(sentence2_list[start_idx + len(hate_speech):])

            if os_set_len < max_row_number :
                new_rows['sentence1'].append(sentence1)
                new_rows['sentence2'].append(corrupted_sentence2)
                new_rows['score'].append(int(0))
                new_rows['sentence2_language'].append(df.loc[curr_os_set_idx, 'sentence2_language'])
                os_set_len += 1
            
            elif df.loc[curr_os_set_idx, 'score'] == 1 :
                    df.loc[curr_os_set_idx, 'sentence2'] = corrupted_sentence2
                    df.loc[curr_os_set_idx, 'score'] = int(0)
                    nb_1_labels -= 1
                    nb_0_labels += 1
                    init_idx_corrupted.append(curr_os_set_idx)

            nb_0_labels += 1
            
            curr_os_set_idx = (curr_os_set_idx + 1) % init_os_set_len
            while curr_os_set_idx in init_idx_corrupted : curr_os_set_idx = (curr_os_set_idx + 1) % init_os_set_len
            curr_hate_set_idx = (curr_hate_set_idx + 1) % len_hate_set
            steps += 1
        
        self.hateset_metadata[self.HATE_LAST_ID][hate_languages[0]] = curr_hate_set_idx
        # --------------------------------------------------------------------------------
        # The potential additional corrupted rows to add
        valid_indexes = [i for i in range(init_os_set_len) if i not in init_idx_corrupted]
        steps = 1
        while nb_0_labels < nb_1_labels :
            # Adding wrong translations
            sentence1 = str(df.loc[curr_os_set_idx, 'sentence1'])
            random_index = np.random.choice(valid_indexes)
            while random_index == curr_os_set_idx : random_index = np.random.choice(valid_indexes)
            corrupted_sentence2 = str(df.loc[random_index, 'sentence2'])

            if os_set_len < max_row_number :
                new_rows['sentence1'].append(sentence1)
                new_rows['sentence2'].append(corrupted_sentence2)
                new_rows['score'].append(int(0))
                new_rows['sentence2_language'].append(df.loc[curr_os_set_idx, 'sentence2_language'])
                os_set_len += 1
            
            elif df.loc[curr_os_set_idx, 'score'] == 1 :
                    df.loc[curr_os_set_idx, 'sentence2'] = corrupted_sentence2
                    df.loc[curr_os_set_idx, 'score'] = int(0)
                    nb_1_labels -= 1
                    nb_0_labels += 1

            curr_os_set_idx = (curr_os_set_idx + 1) % init_os_set_len
            while curr_os_set_idx in init_idx_corrupted : curr_os_set_idx = (curr_os_set_idx + 1) % init_os_set_len
   
        
        new_rows_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_rows_df], ignore_index=True)
        
        return df


    # ---------------------------------------------- HATE SPEECH DATASETS -----------------------------------------------------#

    def load_multilingual_hatespeech_datasets(self) -> None :
        print("LOADING OF THE HATEFUL DATASETS")

        # Loading the datasets
        self.load_all_multilingual_datasets()
        self.load_all_monolingual_datasets()


        # Dataset cleaning for data homegeneity
        processor = Processor()
        for key in self.mult_hate_speech.keys() :
            self.mult_hate_speech[key] = self.mult_hate_speech[key].select_columns([self.HATE_COLUMN_NAME])
            working_df = (self.mult_hate_speech[key]).to_pandas()
            if isinstance(working_df, pd.DataFrame) :
                cleaned_working_df = processor.cleaning(working_df)
                cleaned_working_df[self.HATE_COLUMN_NAME] = cleaned_working_df[self.HATE_COLUMN_NAME].str.replace(r'\buser\b', '', case=False, regex=True)
                # Final replacement
                df = cleaned_working_df.loc[:, :]
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)

                self.mult_hate_speech[key] = df
            
            else : print(f"Error with data type for key : {key}. Continue...")

        # Saving concatenated datasets
        for language in self.mult_hate_speech.keys() :
            df = self.mult_hate_speech[language]
            if isinstance(df, pd.DataFrame) :
                df = df.sample(frac=1, random_state=np.random.randint(5_000)).reset_index(drop=True)
                self.PROCESSED_HATEFUL_DATASET.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.PROCESSED_HATEFUL_DATASET / f'hatefulspeech_{language}.csv', sep = '\t', encoding='utf-8', index=False)

        print("DATASETS LOADED AND SAVED !")
        return
    
    def clean_hateful_speech(self) -> None :
        hateful_sets = os.listdir(self.PROCESSED_HATEFUL_DATASET)
        for file in hateful_sets :
            print(f"File being cleaned : {file}")
            processor = Processor()
            working_df = pd.read_csv(self.PROCESSED_HATEFUL_DATASET / file, encoding='utf-8', sep='\t')

            cleaned_working_df = processor.cleaning(working_df)
            cleaned_working_df[self.HATE_COLUMN_NAME] = cleaned_working_df[self.HATE_COLUMN_NAME].str.replace(r'\buser\b', '', case=False, regex=True)
            # Final replacement
            df = cleaned_working_df.loc[:, :]
            df.replace('', np.nan, inplace=True)
            df.dropna(inplace=True)

            # Saving the file
            self.PROCESSED_HATEFUL_DATASET.mkdir(parents=True, exist_ok=True)
            df.to_csv(self.PROCESSED_HATEFUL_DATASET / file, sep = '\t', encoding='utf-8', index=False)

        return
    
    # ------------ METADATA FUNCTIONS ------------------- #
    def reset_corrupted_files(self) -> None :
        self.metadata_02[self.CORRUPTED_FILES] = []
        with open(self.METADATA_FILE_02, "w", encoding='utf-8') as f :
            json.dump(self.metadata_02, f, indent=4)
    
    def reset_last_checked_id_per_hate_set(self) -> None :
        for key in self.hateset_metadata[self.HATE_LAST_ID].keys() :
            self.hateset_metadata[self.HATE_LAST_ID][key] = 0
        with open(self.HATESET_METADATA, 'w', encoding='utf-8') as f :
            json.dump(self.hateset_metadata, f, indent=4)



    # ----------- HELP FUNCTIONS ------------------------ #
    def load_collected_files(self) -> list :
        """Gets all file paths of the dataset into a list

        Returns:
            list: The list of the files
        """
        top_list_dir = os.listdir(self.MERGED_DATASET)
        list_dir = []  # list
        for sub_folder in top_list_dir:
            if not (self.MERGED_DATASET / sub_folder).is_dir(): list_dir.append(self.MERGED_DATASET / sub_folder)
            else :
                sub_files = os.listdir(self.MERGED_DATASET / sub_folder)
                list_dir.extend([self.MERGED_DATASET / sub_folder / sub_file for sub_file in sub_files])
        return list_dir
    
    def load_all_multilingual_datasets(self) -> None :
        # MLMA dataset
        mlma_hate_speech = load_dataset("nedjmaou/MLMA_hate_speech", split="train")
        mlma_hate_speech = mlma_hate_speech.select_columns(["tweet"])
        mlma_hate_speech = mlma_hate_speech.rename_column("tweet", self.HATE_COLUMN_NAME)

        if isinstance(mlma_hate_speech, Dataset) : self.update_hatespeech_dataset(["fr", "ar", "en"], [mlma_hate_speech])
        else : print("Error with MLMA dataset")

        # OffensEval2020
        for config in ["ar", "da", "gr", "tr"] :
            offenseval_2020 = load_dataset("strombergnlp/offenseval_2020", config, trust_remote_code=True)
            if isinstance(offenseval_2020, DatasetDict) :
                for split in offenseval_2020.keys() :
                    try :
                        dataset = offenseval_2020[split].select_columns(["text"])
                        dataset = offenseval_2020[split].rename_column("text", self.HATE_COLUMN_NAME)
                        self.update_hatespeech_dataset([config], [dataset])
                    except Exception as e :
                        print(f"Error for split {split} : {e}")
                        continue
            else : print(f"Error loading OffensEval2020 dataset")

        

        # CONAN
        df = pd.read_csv(self.HATEFUL_DATASET / "italian_french_english_CONAN.csv", delimiter=",", encoding='utf-8')
        df = df[["hateSpeech"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("hateSpeech", self.HATE_COLUMN_NAME)
        self.update_hatespeech_dataset(["it", "en", "fr"], [dataset])

        # Russian and Ukrainian dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "ukrainian_data.csv", delimiter=",")
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        self.update_hatespeech_dataset(["uk", "ru"], [dataset])

        return
    
    def load_all_monolingual_datasets(self) -> None :
        self.load_albanian_datasets()
        self.load_arabic_datasets()
        self.load_bengali_datasets()
        self.load_chinese_datasets()
        self.load_dutch_datasets()
        self.load_english_datasets()
        self.load_german_datasets()
        self.load_hindi_datasets()
        self.load_indonesian_datasets()
        self.load_italian_datasets()
        self.load_korean_datasets()
        self.load_latvian_datasets()
        self.load_portuguese_datasets()
        self.load_russian_datasets()
        self.load_spanish_datasets()
        self.load_ukrainian_datasets()
        self.load_urdu_datasets()
    
    def load_albanian_datasets(self) -> None :
        datasets = []

        # Albanian hate speech set
        df = pd.read_csv(self.HATEFUL_DATASET / "albanian_hate_speech.csv", delimiter=';', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["sq"], datasets)

    
    def load_arabic_datasets(self) -> None :
        datasets = []

        # L-HASB dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "arabic-L-HSAB.txt", delimiter='\t', encoding='utf-8')
        df = df[["Tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "Tweet" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("Tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Let-Mi dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "arabic-Let-Mi.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["ar"], datasets)
        return
    
    def load_bengali_datasets(self) -> None :
        datasets = []

        # Bengali hate speech dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "bengali_hate_speech.csv", delimiter=',', encoding='utf-8')
        df = df[["sentence"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "sentence" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("sentence", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["bn"], datasets)

        return
    
    def load_chinese_datasets(self) -> None :
        datasets = []

        # Chinese Sexist Comments
        df = pd.read_csv(self.HATEFUL_DATASET / "chinese-SexComment.csv", delimiter=',', encoding='utf-8')
        df = df[["comment_text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "comment_text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("comment_text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["zh"], datasets)

        return    
    
    def load_dutch_datasets(self) -> None :
        datasets = []

        # Dutch hate check
        df = pd.read_csv(self.HATEFUL_DATASET / "dutch-hatecheck.csv", delimiter=',', encoding='utf-8')
        df = df[["test_case"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "test_case" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("test_case", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["nl"], datasets)

        return
        
    def load_english_datasets(self) -> None :
        datasets = []

        # GAB
        df = pd.read_csv(self.HATEFUL_DATASET / "english_gab.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        df.loc[:,"text"] = df.loc[:,"text"].str.replace('1. ', '', regex=True)
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Hasoc 2019 train
        df = pd.read_csv(self.HATEFUL_DATASET / "english_hasoc2019_train.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Hasoc 2019 test
        df = pd.read_csv(self.HATEFUL_DATASET / "english_hasoc2019_test.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Olid train
        df = pd.read_csv(self.HATEFUL_DATASET / "english_olid_train.tsv", delimiter='\t', encoding='utf-8')
        df = df[["tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "tweet" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Olid test
        df = pd.read_csv(self.HATEFUL_DATASET / "english_olid_test.tsv", delimiter='\t', encoding='utf-8')
        df = df[["tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "tweet" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # TBO train
        df = pd.read_excel(self.HATEFUL_DATASET / "english_TBO_train.xlsx")
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Dynamic hatespeech set
        df = pd.read_csv(self.HATEFUL_DATASET / "english-dynamic-hate-speech.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Multitarget CONAN
        df = pd.read_csv(self.HATEFUL_DATASET / "english-Multitarget-CONAN.csv", delimiter=',', encoding='utf-8')
        df = df[["HATE_SPEECH"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "HATE_SPEECH" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("HATE_SPEECH", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Measuring Hate Speech
        measuring_hate_speech = load_dataset("ucberkeley-dlab/measuring-hate-speech")
        if isinstance(measuring_hate_speech, DatasetDict) :
            for split in measuring_hate_speech.keys() :
                dataset = measuring_hate_speech[split]
                dataset = dataset.select_columns(["text"])
                if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
                datasets.append(dataset)

        self.update_hatespeech_dataset(["en"], datasets)

        return
        
    def load_german_datasets(self) -> None :
        datasets = []

        # GermEval2018
        df = pd.read_csv(self.HATEFUL_DATASET / "german_germeval2018_training.txt", delimiter='\t', encoding='utf-8', header=None, names=["sentence", "feature_1", "feature_2"])
        df = df[["sentence"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "sentence" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("sentence", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # HASOC 2019 train
        df = pd.read_csv(self.HATEFUL_DATASET / "german_hasoc2019_train.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # HASOC 2019 test
        df = pd.read_csv(self.HATEFUL_DATASET / "german_hasoc2019_test.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # GAHD
        df = pd.read_csv(self.HATEFUL_DATASET / "german-gahd.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)


        self.update_hatespeech_dataset(["de"], datasets)
        return
        
    def load_hindi_datasets(self) -> None :
        datasets = []

        # HASOC 2019
        hasoc_sets = ["hindi_hasoc2019_train.tsv", "hindi_hasoc2019_test.tsv"]

        for file in hasoc_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter='\t', encoding='utf-8')
            df = df[["text"]].drop_duplicates()
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
            datasets.append(dataset)

        # Codalab
        codalab_sets = ["hindi-codalab-train.csv", "hindi-codalab-eval.csv", "hindi-codalab-test.csv"]
        
        for file in codalab_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter=',', encoding='utf-8')
            df = df[["Post"]].drop_duplicates()
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "Post" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("Post", self.HATE_COLUMN_NAME)
            datasets.append(dataset)

        self.update_hatespeech_dataset(["hi"], datasets)
        return
        
    def load_indonesian_datasets(self) -> None :
        datasets = []

        # RE datasets
        re_sets = ["indonesian_re_dataset_incivility.csv", "indonesian_re_dataset.csv"]
        
        for file in re_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter=',', encoding='utf-8', on_bad_lines='skip')
            df = df[["Tweet"]].drop_duplicates()
            df.loc[:,"Tweet"] = df.loc[:,"Tweet"].str.replace('RT USER:', '', regex=True)
            df.loc[:,"Tweet"] = df.loc[:,"Tweet"].str.replace('USER', '', regex=True)
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "Tweet" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("Tweet", self.HATE_COLUMN_NAME)
            datasets.append(dataset)

        self.update_hatespeech_dataset(["id"], datasets)
        return

    def load_italian_datasets(self) -> None :
        datasets = []

        # Evalita2020e
        evalita2020 = load_dataset("basilepp19/evalita2020-AH-instr")
        if isinstance(evalita2020, DatasetDict) :
            for split in evalita2020.keys() :
                dataset = evalita2020[split]
                dataset = dataset.select_columns(["input"])
                if "input" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("input", self.HATE_COLUMN_NAME)
                datasets.append(dataset)
        
        self.update_hatespeech_dataset(["it"], datasets)
    
        return
    
    def load_korean_datasets(self) -> None :
        datasets = []

        # BEEP
        beep_sets = ["korean_beep_train.tsv", "korean_beep_dev.tsv", "korean_beep_test.tsv"]

        for file in beep_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter='\t', encoding='utf-8')
            df = df[["comments"]].drop_duplicates()
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "comments" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("comments", self.HATE_COLUMN_NAME)
            datasets.append(dataset)
        
        self.update_hatespeech_dataset(["ko"], datasets)
        return
    
    def load_latvian_datasets(self) -> None :
        datasets = []

        # latvian comments
        latvian_sets = ["lituanien-letton-comments-2019.csv"]
        
        for file in latvian_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter='\t', encoding='utf-8')
            df = df[["content"]].drop_duplicates()
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "content" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("content", self.HATE_COLUMN_NAME)
            datasets.append(dataset)

        self.update_hatespeech_dataset(["lv"], datasets)
        return
        
    def load_portuguese_datasets(self) -> None :
        datasets = []

        # TOLD-BR
        df = pd.read_csv(self.HATEFUL_DATASET / "portuguese-ToLD-BR.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # OLID-BR
        olid_br = load_dataset("dougtrajano/olid-br")
        if isinstance(olid_br, DatasetDict) :
            for split in olid_br.keys() :
                dataset = olid_br[split]
                dataset = dataset.select_columns(["text"])
                if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
                datasets.append(dataset)

        self.update_hatespeech_dataset(["pt"], datasets)

        return    

    def load_russian_datasets(self) -> None :
        datasets = []

        # South Park
        df = pd.read_csv(self.HATEFUL_DATASET / "russian_south_park.csv", delimiter=';', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Distorted toxicity
        df = pd.read_csv(self.HATEFUL_DATASET / "russian_distorted_toxicity.tsv", delimiter='\t', encoding='utf-8')
        df = df[["comments"]].drop_duplicates()
        df.loc[:, "comments"] = df.loc[:, "comments"].str.replace(r'^\[id\d+\|[^\]]+\]\s*' + ', ', '', regex=True)
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "comments" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("comments", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["ru"], datasets)
        return    

    def load_spanish_datasets(self) -> None :
        datasets = []

        # Counter hate speech
        counter_hate_speech_es = load_dataset("edumunozsala/counter-hate-speech-es")
        if isinstance(counter_hate_speech_es, DatasetDict) :
            for split in counter_hate_speech_es.keys() :

                ### hatespeech
                dataset = counter_hate_speech_es[split]
                dataset = dataset.select_columns(["HS"])
                if "HS" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("HS", self.HATE_COLUMN_NAME)
                datasets.append(dataset)

                ### counterspeech
                dataset = counter_hate_speech_es[split]
                dataset = dataset.select_columns(["CN"])
                if "CN" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("CN", self.HATE_COLUMN_NAME)
                datasets.append(dataset)

        self.update_hatespeech_dataset(["es"], datasets)


        return    

    def load_ukrainian_datasets(self) -> None :
        datasets = []

        # Ukrainian data
        df = pd.read_csv(self.HATEFUL_DATASET / "ukrainian_data.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["uk"], datasets)
        return    

    def load_urdu_datasets(self) -> None :
        datasets = []

        # Urdu tasks
        tasks_sets = ["urdu_task_1_train.tsv", "urdu_task_1_test.tsv", "urdu_task_1_validation.tsv",
                      "urdu_task_2_train.tsv", "urdu_task_2_test.tsv", "urdu_task_2_validation.tsv",]
        
        for file in tasks_sets :
            df = pd.read_csv(self.HATEFUL_DATASET / file, delimiter='\t', encoding='utf-8', header=None, names=["text", "feature_1"])
            df = df[["text"]].drop_duplicates()
            dataset = Dataset.from_pandas(df.reset_index(drop=True))
            if "text" != self.HATE_COLUMN_NAME : dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
            datasets.append(dataset)

        self.update_hatespeech_dataset(["ur"], datasets)

        
        return
    
    def update_hatespeech_dataset(self, languages: list[str], datasets: list[Dataset]) -> None :
        for language in languages :
            if language in self.mult_hate_speech.keys() :
                datasets.append(self.mult_hate_speech[language])
                self.mult_hate_speech[language] = concatenate_datasets(datasets)
            else :
                self.mult_hate_speech[language] = concatenate_datasets(datasets)
        return