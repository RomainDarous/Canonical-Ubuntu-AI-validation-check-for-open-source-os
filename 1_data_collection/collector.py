import io
from pathlib import Path
import sys
import requests
import zipfile
import pandas as pd
import os
from datetime import datetime
import time
from nordvpn_switcher import initialize_VPN, rotate_VPN, terminate_VPN
from bs4 import BeautifulSoup
import tarfile
import json
import traceback
import numpy as np

class Collector:
    """A class used to collect translations from Weblate and Ubuntu Launchpad.

    Attributes:
        OS_FOLDER (Path): Path to the folder containing OS translations.
        UBUNTU_FOLDER (Path): Path to the folder containing Ubuntu translations.
        OS_DATASET (Path): Path to the dataset folder for OS translations.
        UBUNTU_DATASET (Path): Path to the dataset folder for Ubuntu translations.
        TIME_FORMAT (str): Format for timestamps.
        AWARE (str): Timezone awareness format.
        METADATA_FILE (Path): Path to the metadata file.
        UPDATED_FILES (str): Key for updated files in metadata.
        ARCH_VERSIONS (str): Key for archive versions in metadata.
        VALID_LANGUAGES (str): Key for the list of accepted languages for translations in metadata.
        ALL_FILES (str): Key for the dictionary that maps every file name to its last updated time in metadata.
    """

    # Constants
    OS_FOLDER = Path('./os_by_language')
    UBUNTU_FOLDER = OS_FOLDER

    OS_DATASET = OS_FOLDER / 'dataset'
    UBUNTU_DATASET = OS_DATASET

    # Metadata of the dataset
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    AWARE = '%z'
    METADATA_FILE = Path('./os_by_language/metadata_01.json')
    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"
    VALID_LANGUAGES = "languages"
    WEBLATE_STOP_PROJECT = "weblate_stop"
    LAUNCHPAD_STOP_PROJECT = "launchpad_stop"
    ALL_FILES = "all_files"

    def __init__(self) -> None:
        """
        Initializes the Collector instance and loads metadata from the metadata file.
        """

        self.metadata = {}

        # Assuming the metadata file already has the good format
        try :
            with open(self.METADATA_FILE, 'r', encoding='utf-8') as f :
                self.metadata = json.load(f)
            
        except Exception as e :
            print("Error while loading the metadata file. Try again.")
            sys.exit()



    #---------------------------------- WEBLATE TRANSLATIONS ------------------------------------#

    def get_all_translations(self, api_url: str, api_key: str, wanted_os: list) -> None :
        """Retrieves all translations of a list of Weblate projects.


        Args:
            api_url (str): The URL of the Weblate API.
            api_key (str): Token to access the API, None if missing.
            wanted_os (list): List of project names, or words that must be in the name of the project.

        Returns:
            None
        """   

        try :
            # Headers for authentication
            headers = {
                "Content-Type": "application/json"
            }
            if api_key != "" : headers["Authorization"] = f"Token {api_key}"

            # Connecting to VPN and to the API
            initialize_VPN(save = 1, area_input=['random countries europe 10'])
            url = f"{api_url}projects/"
            projects_dict = self.get_dict(url, headers=headers)

            passed = False
            while True :
                projects = projects_dict['results']

                # Downloading all translations
                for project in projects :

                    # Resume data collection if stopped unexpectidly, to the last opened project
                    if not passed and self.metadata[self.WEBLATE_STOP_PROJECT] and project["slug"] not in self.metadata[self.WEBLATE_STOP_PROJECT] : continue
                    passed = True

                    os_name, is_wanted = self.is_in_wanted_projects(project, wanted_os)
                    if is_wanted : 
                        languages = self.get_dict(project["languages_url"], headers = headers)

                        # Save the file locally
                        download_directory = self.OS_DATASET / os_name
                        download_directory.mkdir(parents=True, exist_ok=True)

                        path = Path(download_directory / project["slug"])
                        self.update_file(languages, project["slug"], path)
                            
                # Terminating or updating API page
                if projects_dict["next"] == None : break
                else :
                    print("\n", projects_dict["next"])
                    projects_dict = self.get_dict(projects_dict["next"], headers=headers)
            
            # Resetting the stop file in metadata to empty as everything went wel
            self.metadata[self.WEBLATE_STOP_PROJECT] = ""

        except KeyboardInterrupt :
            print("Program interrupted by user.")

        except Exception as e :
            print(f"An error has occured : {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)

        finally :
            terminate_VPN()
            # Saving the metadatas in the file
            with open(self.METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4)

    # --------------- WEBLATE HELP FUNCTIONS -------------------------------------------- #

    def update_file(self, languages: dict, project: str, path: Path) -> None :
        """Checks if the language of the current project is downloaded or needs an update.

        Args:
            languages (dict): Dict of the languages dict for a given Weblate project.
            project (str): The slug name of the project being checked.
            path (Path): The path where the translation files should be downloaded.

        Returns:
            dict : Dictionary, keys : updates paths, values : languages to update.
        """

        response = None
        print("Project being checked : ", project)

        for language in languages['results'] :
            update = False
            code = language["code"]
            code_path = path.with_name(f'{path.stem}-{code}.csv')

            # Checking if we want the language
            if code not in self.metadata[self.VALID_LANGUAGES] : continue

            # Getting the update date
            last_dt = datetime.now()
            last_change = language["last_change"]

            ### Trying several time formats to load the last update time
            time_formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M:%S',
                ]
            
            for time_format in time_formats :
                try : 
                    last_dt = datetime.strptime(last_change, time_format)
                    break
                except : continue

            # Checking if another version of the language is in the dataset
            if os.path.exists(code_path) :
                try : 
                    current_dt = datetime.strptime(self.metadata[self.ALL_FILES][str(code_path)], self.TIME_FORMAT)
                    
                    ### Comparing dates
                    if current_dt >= last_dt : 
                        continue
                    else : 
                        print(f"{code_path} : update required")
                        update = True
                        
                except :
                    print("File not found in metadata, downloading again")

                
        
            # Downloading the csv files of the translations
            language_url = language["url"]
            download_url = language_url.replace("projects", "download") + "?format=zip:csv"
            response = requests.get(download_url)

            if response.status_code != 200:
                print(f"Failed to download the file. Status code: {response.status_code}")
                return
            
            # Opening the files of the language
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                files = zip_ref.namelist()
                files.sort()
                trans_df = pd.DataFrame()

                code_df = pd.Series([])
                en_df = pd.Series([])

                for file in files :
                    f = zip_ref.read(file)
                    tmp_df = pd.read_csv(io.StringIO(f.decode('utf-8')))
                    tmp_df = tmp_df.replace('', np.nan).dropna(subset=["source", "target"])
                    if tmp_df.empty :
                        #print(f"{file} : {code} translation empty")
                        continue
                    code_df = pd.concat([code_df,tmp_df['target']], ignore_index=True, axis = 0)
                    en_df = pd.concat([en_df,tmp_df['source']], ignore_index=True, axis = 0)

                ### Saving the translation
                if len(code_df) == 1 : continue
                trans_df["en"] = en_df
                trans_df[code] = code_df
                trans_df.to_csv(code_path, encoding='utf-8', index=False)
                
                ### Saving the updated/created files and their last update date
                self.metadata[self.ALL_FILES][str(code_path)] = last_dt.strftime(self.TIME_FORMAT)
                self.metadata[self.UPDATED_FILES].append(str(code_path))
                self.metadata[self.WEBLATE_STOP_PROJECT] = str(code_path)

                ### Outputing information in the console
                if update : print(f"Updated : {code_path}")
                else : print(f"Downloaded : {code_path}")

        return

    
    def get_dict(self, url: str, headers: dict) -> dict :
        """Checks if a http requests is successful. Otherwise, changes the VPN and tries again.

        Args:
            url (str): URL to access.
            headers (dict): Headers of the url request.

        Returns:
            dict: The json response of the request reponse, empty dict if couldn't connect.
        """

        response = requests.get(url, headers=headers)
        i = 0

        # Check if the request was successful
        while response.status_code != 200 and i < 50:
            print(f"Request failed: {response.status_code} - {response.text}")
            if response.status_code == 429 :
                print("Too many requests, changing VPN")
                rotate_VPN()
                response = requests.get(url, headers=headers)
            i += 1

        # Checking that VPN adress has been changed
        if i < 50 and type(response.json()) == dict: return response.json()
        elif i < 50 and type(response.json()) == list :
            tmp_dict = {}
            tmp_dict['results'] = response.json()
            return tmp_dict
        else : return {}

    def is_in_wanted_projects(self, project: dict, wanted_os: list) -> tuple :
        """Checks whether or not the current project is part of the wanted projects.

        Args:
            project (dict): The current project.
            wanted_projects (list): List of project names that want to be saved, or words that their name must contain.

        Returns:
            tuple: True if the project is in the wanted projects, False otherwise.
        """

        if 'ALL' in wanted_os[0] :
            return wanted_os[0].split(';')[1], True    

        for os_name in wanted_os :
            if os_name.lower() in project["slug"].lower() : return os_name.lower(), True
        return None, False

    #------------------------- UBUNTU TRANSLATIONS on LAUNCHPAD ------------------------------- #

    def get_ubuntu_translation(self, project: str, url: str) -> None :
        """Checks if Ubuntu translations have been updated and/or must be updated.

        Args:
            url (str): the URL of the download page.
            project (str): string to add to every output file

        Returns:
            None
        """

        try :
            # HTTP request to Launchpad
            response = requests.get(url)
            if response.status_code != 200 :
                print(f"Failed to access Launchpad. Status code: {response.status_code}")
                return
            
            # Getting the download link
            soup = BeautifulSoup(response.text, 'html.parser')
            tar_url_component = soup.find_all('a', {'class': 'sprite download'})[0]
            tar_url = tar_url_component.get('href')

            # Getting last update date
            tar_url_text = tar_url_component.text.replace(" ", "").replace('\n', '').replace("UTC", "")
            tar_url_date = datetime.strptime(tar_url_text, '%Y-%m-%d%H:%M:%S')
            

            # Define the download directory
            archive_directory = self.UBUNTU_FOLDER
            archive_directory.mkdir(parents=True, exist_ok=True)


            ### Get the name of the archive
            archive_name = tar_url.split('/')[-1]
            print(f"Archive name: {archive_name}")

            # Checking if the zip file has been updated
            new_zip_file = self.update_zip_file(archive_name, tar_url_date)
            if not new_zip_file : return

            # Define the path to the archive
            archive_path = archive_directory / archive_name

            ### Downloading archive if required
            if archive_path.exists():
                print(f"{archive_name} already downloaded. \nCAUTION : ASSUMING IT'S THE LAST VERSION OF THE ARCHIVE")
            else:
                response = requests.get(tar_url)
                if response.status_code != 200 :
                    print(f"Failed to download the file. Status code: {response.status_code}")
                    return
                with open(archive_path, 'wb') as f:
                    f.write(response.content)
                print(f"{archive_path} downloaded.")

            ### Save the last update date
            self.metadata[self.ARCH_VERSIONS][archive_name] = tar_url_date.strftime(self.TIME_FORMAT)
            
            ### Extract all files to the output directory, storing them by project
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                translations = tar_ref.getmembers()
                
                passed = False
                for translation in translations :
                    if translation.isdir() :
                        print("Exploring ", translation.name)
                    elif translation.isfile() :

                        # Building file path
                        raw_path = Path(translation.name).with_suffix('')
                        raw_path_list = str(raw_path).split('\\')
                        file_name = Path(raw_path_list[-1])                           
                        
                        # Resuming data collection if stopped unexpectidly
                        if not passed and self.metadata[self.LAUNCHPAD_STOP_PROJECT] and file_name != self.metadata[self.LAUNCHPAD_STOP_PROJECT] : continue
                        passed = True

                        try :
                            code = raw_path_list[1]
                            if code not in self.metadata[self.VALID_LANGUAGES] : continue
                            path = self.UBUNTU_DATASET / project / file_name
                            path.parent.mkdir(parents=True, exist_ok=True) 

                        except :
                            print(f"Error for {raw_path}, check what happened !")
                            continue

                        # Opening the file
                        f = tar_ref.extractfile(translation)
                        if f is None : continue

                        content = f.read()
                        self.po_to_csv(content.decode('utf-8').split('\n'), path, code)
                        self.metadata[self.LAUNCHPAD_STOP_PROJECT] = str(file_name)


            ### Delete the archive
            os.remove(archive_path)
            # Updating the metadata file for preprocessing update
            self.metadata[self.UPDATED_FILES].append("ALL;ubuntu-noble")
            self.metadata[self.LAUNCHPAD_STOP_PROJECT] = ""
            
        except KeyboardInterrupt :
            print("Program interrupted by the user")

        except Exception as e :
            print(f"An error has occured : {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type, exc_value, exc_traceback)
        
        finally :
            ### Saving the metadatas in the file
            with open(self.METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=4)


    # ---------------------- LAUNCHPAD HELP FUNCTIONS ---------------------------- #
    def po_to_csv(self, po_content: list, path: Path, code: str) -> bool :
        """Converts a .po file to a .csv file.

        Args:
            po_content (list): .po file content as a list of strings.
            path (Path): destination .csv file.
            code: the ISO code of the language.

        Returns:
            bool: True if the file has been created/updated, False otherwise.
        """    
        # The translation in the target language
        tmp_translation = []
        tmp_en = []
        code_path = path.with_name(f"{path.stem}-{code}.csv")

        # Update check
        creat_date = datetime.now()
        empty = True

        for i, line in enumerate(po_content) :
            # Saving translations
            if "msgid" in line :
                try :
                    id_list = po_content[i].split('"')
                    target_list = po_content[i+1].split('"')
                    id = (f'"{' '.join(id_list[1:])}"')
                    target = (f'"{' '.join(target_list[1:])}"')
                    if len(target) > 2 : empty = False
                    tmp_translation.append(target)
                    tmp_en.append(id)                  
                except :
                    continue
        
        # Building a new csv file to save
        if empty :
            print(f"{code_path} is empty.")
            return False
        else :
            if os.path.exists(code_path) : df = pd.read_csv(code_path)
            else : df = pd.DataFrame()
            tmp_series_for = pd.Series(tmp_translation)
            tmp_series_en = pd.Series(tmp_en)
            df["en"] = tmp_series_en.reset_index(drop=True)
            df[code] = tmp_series_for.reset_index(drop=True)
            df.to_csv(code_path, encoding='utf-8', index=False)
            return True



    def update_zip_file(self, archive_name: str, tar_url_date: datetime) -> bool :
        """Checks if the zip file needs to be updated based on the archive name and date.

        Args :
            archive_name (str): Name of the archive.
            tar_url_date (datetime): Date of the archive URL.

        Returns :
            bool: True if the zip file needs to be updated, False otherwise.
        """
        if archive_name in self.metadata[self.ARCH_VERSIONS] :
            last_update = self.metadata[self.ARCH_VERSIONS][archive_name]
            last_update = datetime.strptime(last_update, self.TIME_FORMAT)

            if last_update >= tar_url_date :
                print("Translation already up-to-date.")
                return False
            else :
                print("Translation needs an update.")
                return True
            
        else : return True

    #---------------------------- METADA MANAGEMENT ---------------------------------#
    def empty_updated_files(self) -> None :
        """Empties the list of updated files in the metadata and saves it to the metadata file."""

        self.metadata[self.UPDATED_FILES] = []
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)

    def empty_archive_files(self) -> None :
        """Empties the archive versions in the metadata and saves it to the metadata file."""

        self.metadata[self.ARCH_VERSIONS] = {}
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)
    
    def empty_all_files(self) -> None :
        """Empties the list of all files in the metadata and saves it to the metadata file."""

        self.metadata[self.ALL_FILES] = {}
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)


