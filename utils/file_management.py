import yaml
import pickle
from config.paths import ARTIFACTS_PATH
import json
import os


def read_yaml(file_name):
    """
    Read and parse a YAML file.

    Parameters:
    - file_name (str): The name of the YAML file to be read.

    Returns:
    - dict: A dictionary containing the parsed YAML content.
    - None: Returns None if an error occurs during file reading or parsing.
    """
    try:
        with open(file_name, 'r') as file:
            yaml_content = yaml.safe_load(file)
            return yaml_content
    except FileNotFoundError:
        print(f"Error: The file {file_name} was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error while reading the YAML file: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    
    
def save_pkl(data, filepath):
    """
    Guarda un objeto en un archivo utilizando pickle.

    Parameters:
    - data: El objeto que se va a guardar.
    - filepath: La ruta donde se guardará el archivo.

    Returns:
    - None
    """
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)


def load_pkl(filepath):
    """
    Carga un objeto desde un archivo pickle.

    Parameters:
    - filepath: La ruta del archivo desde el cual se cargará el objeto.

    Returns:
    - El objeto cargado desde el archivo.
    """
    with open(filepath, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data



def save_json(data, filename):

    """
    Save the JSON data to a file.

    Args:
        data: The JSON data as a dictionary.
        filename: The name of the file to save the JSON data to.

    Returns:
        None.
    """

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def read_json(filepath):

    """
    Get the JSON data from a file.

    Args:
        filepath: The path to the JSON file.

    Returns:
        The JSON data as a dictionary.
    """

    with open(filepath, "r") as f:
        data = json.load(f)
    return data