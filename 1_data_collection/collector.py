import io
from pathlib import Path
import requests
import zipfile
import pandas as pd
import os
from datetime import datetime
from nordvpn_switcher import initialize_VPN, rotate_VPN, terminate_VPN
from bs4 import BeautifulSoup
import tarfile

class Collector:
    """_summary_

    Attributes:
        _type_: _description_
    """

    # Constants
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    OS_FOLDER = Path('./os_by_language')
    UBUNTU_FOLDER = Path('./ubuntu_by_language')
    ZIP_VERSIONS = 'last_zip_versions.txt'
    UPDATED_TRANSLATIONS = 'os_last_updates.txt'

    def __init__(self) -> None:
        pass

    #---------------------------------- WEBLATE TRANSLATIONS ------------------------------------#

    def get_all_translations(self, api_url: str, api_key: str, wanted_projects:list) -> int :
        """Function that gets all translations of a list of Weblate projects


        Args:
            api_url (str): the url of the used api
            api_key (str): token to access to the API, None if missing
            wanted_projects (list): list of project names, or words that must be in the name of the project

        Returns:
            int: 0 to inform the function terminated
        """   

        # Headers for authentication
        headers = {
            "Content-Type": "application/json"
        }
        if api_key != "" : headers["Authorization"] = f"Token {api_key}"

        # Connecting to VPN and to the API
        initialize_VPN(save = 1, area_input=['random countries europe 10'])
        url = f"{api_url}projects/"
        projects_dict = self.get_dict(url, headers=headers)
        stop = False

        while not stop :
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
                        download_directory = self.OS_FOLDER / code
                        added_updated = self.update_file(response, download_directory, last_change)
                        
                        if len(added_updated) != 0:
                            # Updating os_last_update.txt file for preprocessing update
                            with open(self.OS_FOLDER / self.UPDATED_TRANSLATIONS,"a") as f:
                                f.write((datetime.now()).strftime(self.TIME_FORMAT) + "\n")
                                f.writelines(added_updated)
                        
            # Terminating or updating API page
            if projects_dict["next"] == None : break
            else :
                print("\n", projects_dict["next"])
                projects_dict = self.get_dict(projects_dict["next"], headers=headers)

        terminate_VPN()



        return 0

    def get_dict(self, url: str, headers: dict) -> dict :
        """Checks if a http requests is successful. Otherwise, changes the VPN and try again

        Args:
            url (str): url we want to access to
            headers (dict): headers of the url request

        Returns:
            dict: the json file of the request reponse, empty dict if couldn't connect
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
        """Check if the language of the current project is downloaded / needs an update

        Args:
            response (requests.Response): the response of the HTTP request used to get the zip file of the translations
            download_directory (Path): the path where the translation files should be downloaded
            last_change (str): last update for a given project and language (for all components of the project)

        Returns:
            list[str]: list of added or updated files
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
                    prev_dt = datetime.strptime(df["last-update"][0], self.TIME_FORMAT)
                    if prev_dt >= last_dt : print(f"{files} already up-to-date.")
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

                        added_updated.append(str(download_directory / file) + "\n")

                    return added_updated
                        
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
        return []

    def is_in_wanted_projects(self, project: dict, wanted_projects: list) :
        """Checks whether the current project is part of the wanted projects

        Args:
            project (dict): the current project
            wanted_projects (list): list of project names that want to be saved, or words that their name must contain

        Returns:
        None: returns None
        """    

        for project_name in wanted_projects :
            if project_name.lower() in project["name"].lower() : return True
        return False

    #------------------------- UBUNTU TRANSLATIONS on LAUNCHPAD ------------------------------- #

    def get_ubuntu_translation(self, url: str) -> int :
        """Checks if Ubuntu translations have been updated and/or must be updated

        Args:
            url (str): the url of the download page

        Returns:
            int: 0 to confirm the function ended normally, 1 if the translation has not been updated
        """

        # HTTP request to Launchpad
        response = requests.get(url)
        if response.status_code != 200 :
            print(f"Failed to access Launchpad. Status code: {response.status_code}")
            return 1
        
        # Getting the download link
        soup = BeautifulSoup(response.text, 'html.parser')
        tar_url_component = soup.find_all('a', {'class': 'sprite download'})[0]
        tar_url = tar_url_component.get('href')

        # Getting last update date
        tar_url_text = tar_url_component.text.replace(" ", "").replace('\n', '').replace("UTC", "")
        tar_url_date = datetime.strptime(tar_url_text, '%Y-%m-%d%H:%M:%S')
        

        # Define the download directory
        download_directory = self.UBUNTU_FOLDER
        download_directory.mkdir(parents=True, exist_ok=True)


        ### Get the name of the archive
        archive_name = tar_url.split('/')[-1]
        print(f"Archive name: {archive_name}")

        # Checking if the zip file has been updated
        new_zip_file = self.update_zip_file(download_directory, archive_name, tar_url_date, 
                                    'last_zip_downloads.txt')
        if not new_zip_file : return 1

        # Define the path to the archive
        archive_path = download_directory / archive_name

        ### Downloading archive if required
        if archive_path.exists():
            print(f"{archive_name} already downloaded.")
        else:
            response = requests.get(tar_url)
            if response.status_code != 200 :
                print(f"Failed to download the file. Status code: {response.status_code}")
                return 2
            with open(archive_path, 'wb') as f:
                f.write(response.content)
            print(f"{archive_path} downloaded.")

            ##### Save the last update date
            with open(download_directory / self.ZIP_VERSIONS, 'a', encoding='utf-8') as f:
                f.write(f"{archive_name};{tar_url_date.strftime(self.TIME_FORMAT)}")
        
        ### Extract all files to the output directory
        total_added_updated = []
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            translations = tar_ref.getmembers()
            for translation in translations :
                if translation.isdir() :
                    print("Exploring ", translation.name)
                elif translation.isfile() :
                    # Building file path
                    raw_path = Path(translation.name).with_suffix('.csv')
                    raw_path_list = str(raw_path).split('\\')
                    tmp_path = download_directory / Path(f'{raw_path_list[1]}/{raw_path_list[2]}-{raw_path_list[3]}')

                    # Opening the file
                    f = tar_ref.extractfile(translation)
                    if f is None : continue

                    content = f.read()
                    updated = self.po_to_csv(content.decode('utf-8').split('\n'), tmp_path)
                    if updated :
                        # Updating os_last_update.txt file for preprocessing update
                        with open(self.UBUNTU_FOLDER/self.UPDATED_TRANSLATIONS,"a") as f:
                            f.write((datetime.now()).strftime(self.TIME_FORMAT) + "\n")
                            f.write(str(tmp_path) + '\n')
        

        
        """### Delete the archive
        os.remove(archive_path) A REMETTRE"""

        # Reorganizing the dataset
        # TO DO : SAVING OF THE NEW DATE, MOVING CSV files and RENAMING THEM, DELETING ARCHIVE
        # TO DO : AND DELETE SUBFOLDERS AS WELL
        
        return 0


    def po_to_csv(self, po_content: list, csv_path: Path) -> bool :
        """Converts a po file to a csv file

        Args:
            po_content (list): .po file, a list of strings
            csv_path (Path): destination folder for the .csv files

        Returns:
            bool: True if the file has been created/updated, False otherwise
        """    


        tmp_translation = []
        new_date = None
        for i, line in enumerate(po_content) :
            if "PO-Revision-Date:" in line :
                try :
                    line = line.replace(' ', '')
                    tmp_date = line.split(":")[1].split('\n"')[0]
                    tmp_date = datetime.strptime(tmp_date, "%Y-%m-%d %H:%M%z")
                    if os.path.exists(csv_path) :
                        df = pd.read_csv(csv_path)
                        last_update = datetime.strptime(df['last-update'][0], self.TIME_FORMAT)
                        if last_update >= tmp_date : 
                            print(f'{csv_path} already up-to-date.')
                            return False
                        else :
                            new_date = datetime.strftime(tmp_date, self.TIME_FORMAT)
                    else : new_date = datetime.strftime(tmp_date, self.TIME_FORMAT)
                except :
                    continue
            if "msgid" in line :
                try :
                    line = line.replace(' ', '')
                    id = line.split('"')[1]
                    str = po_content[i+1].split('"')[1]
                    if len(id) != 0 and len(str) != 0:
                        tmp_translation.append([id.lower(), str.lower()])
                        
                except :
                    continue
        
        dataframe = pd.DataFrame(tmp_translation, columns=['source', 'target'])
        if dataframe.empty :
            print(f"{csv_path} is empty.")
            return False
        else :
            dataframe['last-update'] = None
            dataframe.at[0, 'last-update'] = new_date
            csv_path.parent.mkdir(parents=True, exist_ok=True) 
            dataframe.to_csv(csv_path, encoding='utf-8', index=False)
            print(f'Created/Updated : {csv_path}')
            return True



    def update_zip_file(self, download_directory, archive_name, tar_url_date, update_file) :
        if (download_directory / update_file).exists() :
            with open(download_directory / update_file, 'r', encoding='utf-8') as f:
                updates = f.readlines()
                for (i,line) in enumerate(updates) :
                    if archive_name in line :
                        last_update = line.split(';')[1]
                        last_update = datetime.strptime(last_update, self.TIME_FORMAT)

                        if last_update >= tar_url_date :
                            print("Translation already up-to-date.")
                            return False
                        else :
                            print("Translation needs an update.")
                            updates.pop(i)
                            return True
        else : return True