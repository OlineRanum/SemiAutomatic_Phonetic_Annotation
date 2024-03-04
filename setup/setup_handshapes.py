import argparse 
import os 
import pandas as pd
import json

def list_subdirectories(directory):
    """
    Returns a list of all subdirectories in the specified directory.

    :param directory: The path of the directory to list subdirectories from.
    :return: A list of subdirectory names.
    """
    # List all entries in the given directory
    entries = os.listdir(directory)
    # Filter out entries that are directories
    subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    return subdirectories

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def write_json_file(file_path, data):
    """Writes data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def filter_data_by_gloss(data, glosses):
    """Filters the data for records where the gloss matches one in the glosses list."""
    filtered_data = []
    for item in data:
        if item.get("gloss") in glosses:
            filtered_data.append(item)
    return filtered_data



if __name__ == "__main__":
    # List  all subdirectories in HE datafolder

    parser = argparse.ArgumentParser(description="Provide configurations for setup")
    parser.add_argument('--datasets', type=str, help='select datasets for preparation',choices=['WLASL', 'SemLex'], default='WLASL')
    parser.add_argument('--directory_path', type=str, help='path to HE directory', default='data/HE_raw_data')
    parser.add_argument('--metadata_file', type=str, help='name of benchmark metadata file', default='data/wlasl_full.json')
    parser.add_argument('--output_file', type=str, help='destination of metadata containing subset corresponding to HE data', default='output/wlasl_subset.json')
    
    
    args = parser.parse_args()

    
    #-----------------------------------------------------------------#
    # 1. Check number of HE glosses available in the specified directory
    #-----------------------------------------------------------------#


    HE_glosses = list_subdirectories(args.directory_path)
    print('------------ Preparing handshapes for the specified datasets ------------')
    if len(HE_glosses) == 0:
        print("No HE glosses found in the specified directory.")
    elif len(HE_glosses) != 1000:
        print("NB: Full dataset not loaded, available glosses is ", len(HE_glosses),' out of 1000 available glosses.\nYou may want to download remaining glosses from the link provided in the README.md file.')
    else:
        print("Number of HE glosses found:", len(HE_glosses))

    #-----------------------------------------------------------------#
    # 2. Setup handshapes for the specified datasets
    #-----------------------------------------------------------------#

    if args.datasets == 'WLASL':
        # Define your list of glosses you want to cross-match
        glosses_to_match = HE_glosses

        def read_json_file(file_path):
            """Reads a JSON file and returns the data."""
            with open(file_path, 'r') as file:
                data = json.load(file)
            return data

        def write_json_file(file_path, data):
            """Writes data to a JSON file."""
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

        def filter_data_by_gloss(data, glosses):
            """Filters the data for records where the gloss matches one in the glosses list."""
            filtered_data = []
            for item in data:
                if item.get("gloss") in glosses:
                    filtered_data.append(item)
            return filtered_data


        # Read the original data
        data = read_json_file(args.metadata_file)
            
        # Filter the data by gloss
        matched_data = filter_data_by_gloss(data, glosses_to_match)
            
        # Write the matched records back to a new JSON file
        write_json_file(args.output_file, matched_data)

        print('\nMatching data for ', args.datasets, ' dataset.')
        print(f"Number of matched glosses: {len(matched_data)}")  
        print(f"Matched data has been written to {args.output_file}")


    if args.datasets == 'SemLex':
        raise NotImplementedError("SemLex dataset is not yet supported. Please select WLASL dataset for setup.")
        """
        semlex = pd.read_csv('../data/semlex_metadata.csv')
        semlex = semlex[semlex['label'].isin(HE_glosses)]

        # Read the list of valid video IDs from poses.txt
        with open('poses.txt', 'r') as file:
            valid_ids = file.read().splitlines()
        semlex = semlex[semlex['video_id'].isin(valid_ids)]
        # Convert the DataFrame into a list of dictionaries based on unique gloss values
        gloss_data = []
        gloss_count = 0
        for gloss, group in semlex.groupby('label'):
            # Create a dictionary for each gloss with its instances
            gloss_count += 1
            gloss_dict = {
                'gloss': gloss,
                'instances': group.drop('label', axis=1).to_dict('records')
            }
            gloss_data.append(gloss_dict)
        """