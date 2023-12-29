import yaml


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
    
    