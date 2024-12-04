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


class Corruptor:

    MERGED_DATASET = Path('./2_os_by_language/datasets')
    METADATA_FILE_02 = Path('./2_os_by_language/metadata_02.json')
    CORRUPTED_FILES = "corrupted_files"
    ROW_NUMBER = "raws_per_file"
    MERGED_DATASET = Path('./2_os_by_language/datasets')
    HATEFUL_DATASET = Path('./multilingual_hateful_sets/data')
    HATE_COLUMN_NAME = "sentence"

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
        except Exception as e:
            print("Error while loading the metadata file. Please try again.")
            sys.exit()

        # Loading hatespeech datasets
        self.mult_hate_speech = DatasetDict()

    def data_corruption(self) -> None :
        print("START OF THE CORRUPTION PROCESS...")
        self.list_dir = list(self.metadata_02[self.ROW_NUMBER].keys())

        # Getting the max number of rows
        max_row_number = max(self.metadata_02[self.ROW_NUMBER].values())

        for file in self.list_dir :
            try :
                language = (file.split('-')[-1]).split('.')[0]
                df = self.corrupt(file, language, max_row_number)
                df.to_csv(file, encoding='utf-8', index=False, sep='\t')
            except Exception as e :
                print(f"Error with file {file} : {e}")
            
        print("DATA SUCCESSFULLY CORRUPTED !")
        return
    
    def corrupt(self, file: str, language: str, max_row_number: int) -> DataFrame :
        print(f"Corrupting : {file}...")
        # Loading the dataset
        df = pd.read_csv(file, encoding='utf-8', delimiter='\t')
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Dataset features
        nb_0_labels = np.sum(df['score'] == 0)
        nb_1_labels = np.sum(df['score'] == 1)
        row_number = df.shape[0]
        curr_idx = 0

        # Hatespeech dataset features
        if language in self.mult_hate_speech.keys() : 
            row_hate_number = len(self.mult_hate_speech[language])
            code = language
        else :
            row_hate_number = len(self.mult_hate_speech["en"])
            code = 'en'

        curr_hate_idx = 0

        # The potential additional corrupted rows to add
        new_rows = {'sentence1' : [],
                    'sentence2' : [],
                    'score' : []}

        
        while nb_0_labels < nb_1_labels :
            sentence1 = str(df.loc[curr_idx, 'sentence1'])
            sentence2 = str(df.loc[curr_idx, 'sentence2'])
            corrupted_sentence2 = sentence2

            hate_speech = (self.mult_hate_speech[code])['sentence'][curr_hate_idx]
            hate_speech = re.sub(r'@\w+\s?', ' ', hate_speech)

            hate_speech_list = hate_speech.split(' ')
            sentence2_list = sentence2.split(' ')

            if len(hate_speech_list) >= len(sentence2_list) :
                corrupted_sentence2 = hate_speech
            else :
                start_idx = np.random.randint(0, len(sentence2_list) - len(hate_speech_list))
                corrupted_sentence2 = ' '.join(sentence2_list[:start_idx]) + hate_speech + ' ' + ' '.join(sentence2_list[start_idx + len(hate_speech):])
                print(corrupted_sentence2)

            if row_number < max_row_number :
                new_rows['sentence1'].append(sentence1)
                new_rows['sentence2'].append(corrupted_sentence2)
                new_rows['score'].append(0)
                row_number += 1
            
            else :
                df.loc[curr_idx, 'sentence2'] = corrupted_sentence2
                df.loc[curr_idx, 'score'] = 0
                nb_1_labels -= 1

            nb_0_labels += 1
            
            curr_idx = (curr_idx + 1) % row_number
            curr_hate_idx = (curr_hate_idx + 1) % row_hate_number
        
        new_rows_df = pd.DataFrame(new_rows)
        
        df = pd.concat([df, new_rows_df], ignore_index=True)
        return df


    # ---------------------------------------------- HATE SPEECH DATASETS -----------------------------------------------------#

    def load_multilingual_hatespeech_datasets(self) -> None :
        print("LOADING OF THE HATEFUL DATASETS")

        # Loading the datasets
        self.load_all_multilingual_datasets()
        self.load_all_monolingual_datasets()

        # Dataset cleaning for data home
        processor = Processor()
        for key in self.mult_hate_speech.keys() :
            working_df = (self.mult_hate_speech[key]).to_pandas()
            if isinstance(working_df, pd.DataFrame) :
                cleaned_working_df = processor.cleaning(working_df)
                
                # Final replacement
                df = cleaned_working_df.loc[:, :]
                df.replace('', np.nan, inplace=True)
                df.dropna(inplace=True)

                self.mult_hate_speech[key] = Dataset.from_pandas(df)
            
            else : print(f"Error with data type for key : {key}. Continue...")

        print("DATASETS LOADED !")
        return

    # From https://huggingface.co/Mike0307/multilingual-e5-language-detection


    def predict(self, text, model, tokenizer, device = torch.device('cpu')):
        model.to(device)
        model.eval()
        tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        return probabilities


    
    # ------------ METADATA FUNCTIONS ------------------- #
    def reset_corrupted_files(self) -> None :
        self.metadata_02[self.CORRUPTED_FILES] = []
        with open(self.metadata_02, "w", encoding='utf-8') as f :
            json.dump(self.metadata_02, f, indent=4)



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
        """
        # MLMA dataset
        mlma_hate_speech = load_dataset("nedjmaou/MLMA_hate_speech", split="train")
        if isinstance(mlma_hate_speech, Dataset) : self.update_hatespeech_dataset(["fr", "ar", "en"], [mlma_hate_speech])
        else : print("Error with MLMA dataset")

        # OffensEval2020
        for config in ["ar", "da", "en", "gr", "tr"] :
            offenseval_2020 = load_dataset("strombergnlp/offenseval_2020", config=config)
            if isinstance(offenseval_2020, DatasetDict) :
                for split in offenseval_2020.keys() :
                    offenseval_2020[split] = (offenseval_2020[split].select_columns(["text"])).rename_column("text", self.HATE_COLUMN_NAME)
                self.update_hatespeech_dataset([config], [offenseval_2020[split] for split in offenseval_2020.keys()])
            else : print(f"Error loading OffensEval2020 dataset")
        """

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
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["sq"], datasets)

    
    def load_arabic_datasets(self) -> None :
        datasets = []

        # L-HASB dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "arabic-L-HSAB.txt", delimiter='\t', encoding='utf-8')
        df = df[["tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Let-Mi dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "arabic-Let-Mi.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["ar"], datasets)
        return
    
    def load_bengali_datasets(self) -> None :
        datasets = []

        # Bengali hate speech dataset
        df = pd.read_csv(self.HATEFUL_DATASET / "bengali_hate_speech.csv", delimiter=',', encoding='utf-8')
        df = df[["sentence"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("sentence", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["bn"], datasets)

        return
    
    def load_chinese_datasets(self) -> None :
        datasets = []

        # Chinese Sexist Comments
        df = pd.read_csv(self.HATEFUL_DATASET / "chinese-SexComment.csv", delimiter=',', encoding='utf-8')
        df = df[["comment_text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("comment_text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        self.update_hatespeech_dataset(["zh"], datasets)

        return    
    
    def load_dutch_datasets(self) -> None :
        datasets = []

        # Dutch hate check
        df = pd.read_csv(self.HATEFUL_DATASET / "dutch-hatecheck.csv", delimiter=',', encoding='utf-8')
        df = df[["test_case"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("test_case", self.HATE_COLUMN_NAME)
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
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Hasoc 2019 train
        df = pd.read_csv(self.HATEFUL_DATASET / "english_hasoc2019_train.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Hasoc 2019 test
        df = pd.read_csv(self.HATEFUL_DATASET / "english_hasoc2019_test.tsv", delimiter='\t', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Olid train
        df = pd.read_csv(self.HATEFUL_DATASET / "english_olid_train.tsv", delimiter='\t', encoding='utf-8')
        df = df[["tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Olid test
        df = pd.read_csv(self.HATEFUL_DATASET / "english_olid_text.tsv", delimiter='\t', encoding='utf-8')
        df = df[["tweet"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("tweet", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # TBO train
        df = pd.read_excel(self.HATEFUL_DATASET / "english_TBO_train.xlsx")
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Dynamic hatespeech set
        df = pd.read_csv(self.HATEFUL_DATASET / "english-dynamic-hate-speech.csv", delimiter=',', encoding='utf-8')
        df = df[["text"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("text", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Multitarget CONAN
        df = pd.read_csv(self.HATEFUL_DATASET / "english-Multitarget-CONAN.csv", delimiter=',', encoding='utf-8')
        df = df[["HATE_SPEECH"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("HATE_SPEECH", self.HATE_COLUMN_NAME)
        datasets.append(dataset)

        # Sexism annotations
        df = pd.read_csv(self.HATEFUL_DATASET / "dutch-hatecheck.csv", delimiter=',', encoding='utf-8')
        df = df[["test_case"]].drop_duplicates()
        dataset = Dataset.from_pandas(df.reset_index(drop=True))
        dataset = dataset.rename_column("test_case", self.HATE_COLUMN_NAME)
        datasets.append(dataset)


        self.update_hatespeech_dataset(["en"], datasets)

        return
        
    def load_german_datasets(self) -> None :

        return
        
    def load_hindi_datasets(self) -> None :
   
        return
        
    def load_indonesian_datasets(self) -> None :

        return

    def load_italian_datasets(self) -> None :
    
        return    
    
    def load_korean_datasets(self) -> None :

        return
    
    def load_latvian_datasets(self) -> None :

        return
        
    def load_portuguese_datasets(self) -> None :

        return    

    def load_russian_datasets(self) -> None :

        return    

    def load_spanish_datasets(self) -> None :

        return    

    def load_ukrainian_datasets(self) -> None :

        return    

    def load_urdu_datasets(self) -> None :

        return
    
    def update_hatespeech_dataset(self, languages: list[str], datasets: list[Dataset]) -> None :
        for language in languages :
            if language in self.mult_hate_speech.keys() :
                datasets.append(self.mult_hate_speech[language])
                self.mult_hate_speech[language] = concatenate_datasets(datasets)
            else :
                self.mult_hate_speech[language] = concatenate_datasets(datasets)
        return
