import io
from pathlib import Path
import sys
import requests
import zipfile
import pandas as pd
import os
from datetime import datetime
from nordvpn_switcher import initialize_VPN, rotate_VPN, terminate_VPN
from bs4 import BeautifulSoup
import tarfile
import json
import traceback

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
    """

    # Constants
    OS_FOLDER = Path('./os_by_language')
    UBUNTU_FOLDER = OS_FOLDER
    OS_DATASET = OS_FOLDER / 'dataset'
    UBUNTU_DATASET = OS_DATASET

    # Metadata of the dataset
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    AWARE = '%z'
    METADATA_FILE = Path('./os_by_language/dataset_metadata.json')
    UPDATED_FILES = "updated_files"
    ARCH_VERSIONS = "archive_versions"

    def __init__(self) -> None:
        """
        Initializes the Collector instance and loads metadata from the metadata file.
        If the metadata file is not found or is empty, initializes metadata with default values.
        """

        self.metadata = {}

        # Assuming the metadata file already has the good format
        try :
            with open(self.METADATA_FILE, 'r', encoding='utf-8') as f :
                self.metadata = json.load(f)
        except :
            print("Error loading the metadata file : empty dict instead")
            self.metadata = {}
        
        if not self.metadata:
            self.metadata = {
                self.UPDATED_FILES : [],
                self.ARCH_VERSIONS : {}
            }

    #---------------------------------- WEBLATE TRANSLATIONS ------------------------------------#

    def get_all_translations(self, api_url: str, api_key: str, wanted_projects: list) -> None :
        """Retrieves all translations of a list of Weblate projects.


        Args:
            api_url (str): The URL of the Weblate API.
            api_key (str): Token to access the API, None if missing.
            wanted_projects (list): List of project names, or words that must be in the name of the project.

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

            while True :
                projects = projects_dict['results']
                # Downloading all translations
                for project in projects :
                    if wanted_projects[0] == "ALL" or self.is_in_wanted_projects(project, wanted_projects) : 
                        languages = self.get_dict(project["languages_url"], headers = headers)

                        for language in languages :
                            code = language["code"]
                            if code == 'en' : continue
                            last_change = language["last_change"]
                            language_url = language["url"]

                            # getting the csv files
                            download_url = language_url.replace("projects", "download") + "?format=zip:csv"
                            response = requests.get(download_url)

                            # Save the file locally
                            download_directory = self.OS_DATASET / code
                            added_or_updated_files = self.update_file(response, download_directory, last_change)
                            
                            if len(added_or_updated_files) != 0:
                                # Updating metadatas
                                self.metadata[self.UPDATED_FILES].extend(added_or_updated_files)
                            
                # Terminating or updating API page
                if projects_dict["next"] == None : break
                else :
                    print("\n", projects_dict["next"])
                    projects_dict = self.get_dict(projects_dict["next"], headers=headers)

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
        if i < 50 : return response.json()
        else : return {}


    def update_file(self, response: requests.Response, download_directory: Path, last_change: str) -> list[str] :
        """Checks if the language of the current project is downloaded or needs an update.

        Args:
            response (requests.Response): The response of the HTTP request used to get the zip file of the translations.
            download_directory (Path): The path where the translation files should be downloaded.
            last_change (str): Last update for a given project and language (for all components of the project).

        Returns:
            list[str]: List of added or updated files.
        """    

        download_directory.mkdir(parents=True, exist_ok=True)

        if response.status_code == 200:
            # Use BytesIO to handle the file in memory
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                # Get file names
                files = zip_ref.namelist()

                path = os.path.join(download_directory, files[0])
                update = False
                print("File being checked : ", path)

                # Trying several time formats
                time_formats = [
                    "%Y-%m-%dT%H:%M:%S.%fZ",
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S',
                    ]
                
                last_dt = datetime.now()

                for time_format in time_formats :
                    try : 
                        last_dt = datetime.strptime(last_change, time_format)
                        break
                    except ValueError : continue

                # Checking if an update is required
                if os.path.exists(path) :
                    df = pd.read_csv(path, encoding='utf-8')
                    current_dt = datetime.strptime(df["last-update"][0], self.TIME_FORMAT)
                    if current_dt >= last_dt : print(f"{files} already up-to-date.")
                    else : update = True

                # Updating the files if required
                if not os.path.exists(path) or update :
                    zip_ref.extractall(download_directory)
                    added_updated = []
                    for file in files :
                        path = os.path.join(download_directory, file) 
                        df = pd.read_csv(path, encoding='utf-8')

                        # Droping empty rows and useless columns
                        df = df.drop([column for column in df.columns if column not in ['target', 'source']], axis = 1)
                        df = df.dropna(subset = ['source', 'target'])
                        
                        # Checking if the file is empty
                        if df.empty :
                            os.remove(path) 
                            print(f"{file} is empty")
                            continue

                        # Droping useless columns and saving the file
                        df["last-update"] = None
                        df.at[0, "last-update"] = last_dt.strftime(self.TIME_FORMAT)
                        df.to_csv(path, encoding='utf-8', index=False)
                        
                        if update : print(f"Updated : {path}")
                        else : print(f"Downloaded : {path}")

                        added_updated.append(str(download_directory / file))

                    return added_updated
                        
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
        return []

    def is_in_wanted_projects(self, project: dict, wanted_projects: list) -> bool :
        """Checks whether or not the current project is part of the wanted projects.

        Args:
            project (dict): The current project.
            wanted_projects (list): List of project names that want to be saved, or words that their name must contain.

        Returns:
            bool: True if the project is in the wanted projects, False otherwise.
        """    

        for project_name in wanted_projects :
            if project_name.lower() in project["name"].lower() : return True
        return False

    #------------------------- UBUNTU TRANSLATIONS on LAUNCHPAD ------------------------------- #

    def get_ubuntu_translation(self, url: str) -> None :
        """Checks if Ubuntu translations have been updated and/or must be updated.

        Args:
            url (str): the URL of the download page.

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
            
            ### Extract all files to the output directory
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                translations = tar_ref.getmembers()
                for translation in translations :
                    if translation.isdir() :
                        #print("Exploring ", translation.name)
                        pass
                    elif translation.isfile() :
                        # Building file path
                        raw_path = Path(translation.name).with_suffix('')
                        raw_path_list = str(raw_path).split('\\')
                        try :
                            tmp_path = self.UBUNTU_DATASET / Path(f'{raw_path_list[1]}/{raw_path_list[2]}-{raw_path_list[3]}-{raw_path_list[1]}.csv')
                        except :
                            print(f"Error for {raw_path_list}, check what happened !")
                            with open("errors.txt", "a") as f :
                                f.write(f"Error for {raw_path_list}, check what happened !\n")
                            continue

                        # Opening the file
                        f = tar_ref.extractfile(translation)
                        if f is None : continue

                        content = f.read()
                        updated = self.po_to_csv(content.decode('utf-8').split('\n'), tmp_path)
                        if updated :
                            # Updating the metadata file for preprocessing update
                            self.metadata[self.UPDATED_FILES].append(str(tmp_path))

            """### Delete the archive
            os.remove(archive_path)"""
            
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


    def po_to_csv(self, po_content: list, csv_path: Path) -> bool :
        """Converts a .po file to a .csv file.

        Args:
            po_content (list): .po file content as a list of strings.
            csv_path (Path): destination .csv file.

        Returns:
            bool: True if the file has been created/updated, False otherwise.
        """    
        # The translation in the target language
        tmp_translation = []

        # Update check
        creat_date = datetime.now()
        rev_date = datetime.now()

        for i, line in enumerate(po_content) :
            # Saving translations
            if "msgid" in line :
                try :
                    id = line.split('"')[1]
                    target = po_content[i+1].split('"')[1]
                    if len(id) != 0 and len(target) != 0:
                        tmp_translation.append([id.lower(), target.lower()])                  
                except :
                    continue

            # Getting Revision date
            elif "PO-Revision-Date" in line :
                
                try :
                    rev_date = line.split(": ")[1][:-3]
                    rev_date = datetime.strptime(rev_date, "%Y-%m-%d %H:%M%z")
                    continue
                    if not os.path.exists(csv_path) : 
                        print("No .csv file found")
                        continue
                    
                    df = pd.read_csv(csv_path)
                    if not df['last-update'][0] or not isinstance(df['last-update'][0], str) : 
                        print("Unable to load the last update time")
                        continue

                    curr_dt = datetime.strptime(df['last-update'][0], self.TIME_FORMAT)
                    
                    if curr_dt >= max(rev_date, creat_date) : 
                        print(f'{csv_path} already up-to-date.')
                        return False
                    
                except Exception as e:
                    print(f"An error occured here : {e}")
                    continue
            
            # Getting Creation date
            elif "POT-Creation-Date" in line :
                try :
                    creat_date = line.split(": ")[1][:-3]
                    creat_date = datetime.strptime(creat_date, "%Y-%m-%d %H:%M%z")
                except :
                    continue

        # Building a new csv file to save
        dataframe = pd.DataFrame(tmp_translation, columns=['source', 'target'])
        if dataframe.empty :
            #print(f"{csv_path} is empty.")
            return False
        else :
            dataframe['last-update'] = None
            dataframe.at[0, 'last-update'] = datetime.strftime(max(rev_date, creat_date), self.TIME_FORMAT + self.AWARE)
            csv_path.parent.mkdir(parents=True, exist_ok=True) 
            dataframe.to_csv(csv_path, encoding='utf-8', index=False)
            #print(f'Created/Updated : {csv_path}')
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

    #---------------- METADA MANAGEMENT ----------------------------#
    def empty_updated_files(self) -> None :
        """Empties the list of updated files in the metadata and saves it to the metadata file.

        Returns :
            None
        """

        self.metadata[self.UPDATED_FILES] = []
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)

    def empty_archive_files(self) -> None :
        """Empties the archive versions in the metadata and saves it to the metadata file.
        
        Returns :
            None
        """

        self.metadata[self.ARCH_VERSIONS] = {}
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)
    
    def reset_metadata(self) -> None :
        """Resets the metadata to default values and saves it to the metadata file.

        Returns : 
            None
        """

        self.metadata = {
            self.UPDATED_FILES : [],
            self.ARCH_VERSIONS : {}
        }
        with open(self.METADATA_FILE, "w", encoding='utf-8') as f :
            json.dump(self.metadata, f, indent=4)

