import os
import re
from collections import defaultdict
import shutil



def group_directories(dir_path, pattern):
    """ Group directories by base name and find the highest number
    """
    # Dictionary to hold the highest version of each base name
    highest_versions = defaultdict(lambda: -1)
    
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):  # Ensure it's a directory
            match = pattern.match(entry)
            if match:
                base_name, number = match.groups()
                number = int(number) if number else 0  # If only one take, set number to 0
                highest_versions[base_name] = max(highest_versions[base_name], number)
    return highest_versions
                

def delete_dir(dir_path, pattern):
    """ Delete directories
    """
    # Find latest version of directory
    highest_versions = group_directories(dir_path, pattern)
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isdir(full_path):  # Ensure it's a directory
            match = pattern.match(entry)
            if match:
                base_name, number = match.groups()
                number = int(number) if number else 0
                if number < highest_versions[base_name]:
                    shutil.rmtree(full_path)  
                    print(f"Deleted: {full_path}")
    print("Cleanup complete.")

def rename_dir(dir_path, pattern, prependix = 'oline_asl_'):
    """ Rename directories
    """
    for dirname in os.listdir(dir_path):
        full_path = os.path.join(dir_path, dirname)
        if os.path.isdir(full_path):  # Ensure it's a directory
            match = pattern.match(dirname)
            if match:
                new_name = match.group(1)  # New directory name without numbers
                new_full_path = os.path.join(dir_path, new_name[len(prependix):]) # Remove Prefix 
                os.rename(full_path, new_full_path)  # Rename the directory
                print(f"Renamed: {dirname} to {new_name[len(prependix):]}")
    print("Renaming of directories complete.")


if __name__ == "__main__":
    # HE directory path
    dir_path = r'../HE_raw_data'

    # Regular expression to match the base name and number number
    pattern = re.compile(r'^(.*?)(?: \((\d+)\))?$')
    delete_dir(dir_path, pattern)
    rename_dir(dir_path, pattern)