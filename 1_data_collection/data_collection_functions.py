import io
from pathlib import Path
import requests
import zipfile
import pandas as pd
import os
from datetime import datetime
from nordvpn_switcher import initialize_VPN, rotate_VPN, terminate_VPN

#---------------------------------- WEBLATE TRANSLATIONS ------------------------------------#

def get_all_translations(api_url: str, api_key: str, wanted_projects:list) -> int :
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
    if api_key != None : headers["Authorization"] = f"Token {api_key}"

    # Connecting to VPN and to the API
    initialize_VPN(save = 1, area_input=['random countries europe 10'])
    url = f"{api_url}projects/"
    projects_dict = get_dict(url, headers=headers)
    stop = False

    while not stop :
        projects = projects_dict['results']
        # Downloading all translations
        for project in projects :
            if wanted_projects == "ALL" or is_in_wanted_projects(project, wanted_projects) : 
                languages = get_dict(project["languages_url"], headers = headers)

                for language in languages :
                    code = language["code"]
                    if code == 'en' : continue
                    last_change = language["last_change"]
                    language_url = language["url"]

                    # getting the csv files
                    download_url = language_url.replace("projects", "download") + "?format=zip:csv"
                    response = requests.get(download_url)

                    # Save the file locally
                    download_directory = Path(f'./os_by_language/{code}')
                    update_file(response, download_directory, last_change)

        # Terminating or updating API page
        if projects_dict["next"] == None : break
        else :
            print("\n", projects_dict["next"])
            projects_dict = get_dict(projects_dict["next"], headers=headers)

    terminate_VPN()
    return 0

def get_dict(url: str, headers: dict) -> dict :
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


def update_file(response: requests.Response, download_directory: Path, last_change: str) -> int :
    """Check if the language of the current project is downloaded / needs an update

    Args:
        response (requests.Response): the response of the HTTP request used to get the zip file of the translations
        download_directory (Path): the path where the translation files should be downloaded
        last_change (str): last update for a given project and language (for all components of the project)

    Returns:
        int: 0 to specify the function ended
    """    

    download_directory.mkdir(parents=True, exist_ok=True)

    if response.status_code == 200:
        # Use BytesIO to handle the file in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            # Extract all files to the output directory

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
                prev_dt = datetime.strptime(df["last-update"][0], '%Y-%m-%d %H:%M:%S.%f')
                if prev_dt >= last_dt : print(f"{files} already up-to-date")
                else : update = True

            # Updating the files if requires
            if not os.path.exists(path) or update :
                zip_ref.extractall(download_directory)
                for file in files :
                    path = os.path.join(download_directory, file) 
                    df = pd.read_csv(path, encoding='utf-8')

                    # Droping empty rows and useless columns
                    df = df.drop(['location', 'id', 'fuzzy', 'context', 'translator_comments', 'developer_comments'], axis = 1)
                    df = df.dropna(subset = ['source', 'target'])
                    

                    # Checking if the file is empty
                    if df.empty :
                        os.remove(path) 
                        print(f"{file} is empty")
                        continue

                    # Droping useless columns and saving the file
                    df["last-update"] = last_dt.strftime('%Y-%m-%d %H:%M:%S.%f')
                    df.to_csv(path, encoding='utf-8', index=False)
                    
                    if update : print(f"Updated : {path}")
                    else : print(f"Downloaded : {path}")
                    
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
    return 0


def is_in_wanted_projects(project: dict, wanted_projects: list) :
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

def get_ubuntu_translation(url: str) -> int :
    """Checks if Ubuntu translations have been updated and/or must be updated

    Args:
        url (str): the url of the download page

    Returns:
        int: 0 to confirm the function ended normally
    """    
    return 0


def po_to_csv(po_path: str, csv_path: str) -> int :
    """Converts a po file to a csv file

    Args:
        po_path (str): path of a .po file
        csv_path (str): path of the .csv file

    Returns:
        int: 0 to confirm the function ended normally
    """    
    po_files = os.listdir(po_path)

    for file in po_files :
        lge = file.split('.')[0]
        tmp_translation = []
        with open(po_path+file, 'r', encoding='utf-8') as content :
            lines = content.readlines()
            for i, line in enumerate(lines) :
                if "msgid" in line :
                    id = line.split('"')[1]
                    str = lines[i+1].split('"')[1]
                    if len(id) != 0 and len(str) != 0:
                        tmp_translation.append([id.lower(), str.lower()])

        dataframe = pd.DataFrame(tmp_translation, columns=['en', lge])
        dataframe.to_csv(csv_path+lge+'.csv', encoding='utf-8')
    return 0

