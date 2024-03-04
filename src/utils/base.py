import json 
import random
import pandas as pd



class BaseAnnotationUtils:
    def __init__(self, hand = "R", 
                 path = 'data/HE_raw_data/'):

        self.hand = hand
        self.main_dir = path
        
        # Selected columns from the stretchsense CSV file
        self.selected_cols = [
            "Timecode(device)", "Timer(device)", "Timecode(master)", "hand_x", "hand_y", "hand_z",
            "index_00_x", "index_00_y", "index_00_z",
            "index_01_x", "index_01_y", "index_01_z",
            "index_02_x", "index_02_y", "index_02_z",
            "index_03_x", "index_03_y", "index_03_z",
            "middle_00_x", "middle_00_y", "middle_00_z",
            "middle_01_x", "middle_01_y", "middle_01_z",
            "middle_02_x", "middle_02_y", "middle_02_z",
            "middle_03_x", "middle_03_y", "middle_03_z",
            "pinky_00_x", "pinky_00_y", "pinky_00_z",
            "pinky_01_x", "pinky_01_y", "pinky_01_z",
            "pinky_02_x", "pinky_02_y", "pinky_02_z",
            "pinky_03_x", "pinky_03_y", "pinky_03_z",
            "ring_00_x", "ring_00_y", "ring_00_z",

            "ring_01_x", "ring_01_y", "ring_01_z",
            "ring_02_x", "ring_02_y", "ring_02_z",
            "ring_03_x", "ring_03_y", "ring_03_z",
            "thumb_01_x", "thumb_01_y", "thumb_01_z",
            "thumb_02_x", "thumb_02_y", "thumb_02_z",
            "thumb_03_x", "thumb_03_y", "thumb_03_z"
            ]
        
        # Count empty data directories
        self.empty_data_counter = 0
        self.empt_dir = 0
    
    def read_calibration(self, file_path):
        """ Reads the calibration data from a JSON file and returns it as a pandas dataframe.
        Used downstream to calculate the Euclidean distance between the hand and the calibration poses.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)

        poses_data = []

        for _, pose_info in data['calibration'][0]['poses'].items():
            pose_name = pose_info['name']
            joints = pose_info['joints'][3:]
            poses_data.append((pose_name, joints))

        df = pd.DataFrame(poses_data, columns=['Pose Name', 'Joints'])
        
        df.set_index('Pose Name', inplace=True)
        droprows = ['Fist', 'Paddle', 'Reach', 'Fist Thumb Out', 'Thumbs Up', 'open_b_tight']
        for row in droprows:
            if row in df.index:
                df = df.drop(index=row)
        
        return df
    
    def read_asllex_handshapes(self, benchmark_metadata = 'output/wlasl_989.json'):
        """ Reads the handshapes from the ASLLex dataset and saves them to a file.
        Also sets the handshapes attribute of the class to the read data.
        """	
        with open(benchmark_metadata, 'r') as file:
            data = json.load(file)

        gloss_handshape_dict = {}
        for entry in data:
            gloss = entry['gloss']

            if entry['instances'] and 'Handshape' in entry['instances'][0]:
                handshape = entry['instances'][0]['Handshape']
                gloss_handshape_dict[gloss] = handshape


        with open('output/original_asllex_handshapes.json', 'w') as outfile:
            json.dump(gloss_handshape_dict, outfile, indent=4)

        print("Data has been processed and saved.")
        self.handshapes = gloss_handshape_dict
       
    def select_new_handshape(self, df, filterratio = 0.1):
        """ Selects the handshape with the highest proportion of occurrences in the dataframe.
        This function selects '5' only if it is detected in more than 90% of the frames.

        """

        value_counts = df['closest_calibration'].value_counts(dropna=False)

        total_count = value_counts.sum()
        proportions = value_counts / total_count

        # Remove values with less than 10% of the total number of frames 
        filtered_proportions = proportions[proportions >= filterratio]
        
        # Special handling for '5', which is the typical resting position of the hand 
        special_values = ['5']

        # Check if special values are present and filter accordingly
        filtered_specials = filtered_proportions[filtered_proportions.index.isin(special_values)]
        
        if not filtered_specials.empty and len(filtered_proportions) > len(filtered_specials):
            # If there are special values and other values, exclude special values
            return filtered_proportions[~filtered_proportions.index.isin(special_values)].idxmax()
        
        elif not filtered_specials.empty:
            # If only special values are present, keep the one with the highest ratio
            # This will only happen if 5 is present in more than 90% of the frames, due to the filtering
            return filtered_specials.idxmax()
        
        else:
            # Return the value with the highest proportion if no special values are present
            return filtered_proportions.idxmax()

    def output_txt(self, glosses, handshapes, filename='output/ED_handshapes.txt'):
        """ Writes glosses and handshapes to a file, with each pair on a new line.
        """
        with open(filename, 'w') as file:
            for gloss, handshape in zip(glosses, handshapes):
                file.write(f"{gloss}, {handshape}\n")
        
        print(f"Data has been written to {filename}.")
        
    def load_sample(self, subdir, file):
        """ Load subdirectory csv into pandas dataframe 
        """
        try:
            df = pd.read_csv(subdir +'/'+ file)

            # Clean column titles 
            # TODO: fix this in all data before publishing 
            df.columns = df.columns.str.replace(' ', '')
            # Load timeseries data into seperate df
            df_timecodes = df[self.selected_cols[0:3]]
            df = df[self.selected_cols[6:]]
            return df, df_timecodes
        
        except Exception as e:
            # If the data is empty, this is usually because no data was recorded for the given glove
            # This might be because no change in position was detected
            # Code goes on to check glove on other hand
            # If both files are empty, a label '5' will be given 
            self.empty_data_counter += 1
            return None, None

    def select_hand(self, lst):
        """ After calculating the handshape label for each hand, select one of them
        as an approximation to the dominant hand.
        Felxible in order to handle that the signer might not enforce the use of their
        dominant hand to perform every sign. 
        """
        
        # If both elements are -1, return '5' and indicate no index (-1 or another signal value could be used)
        if lst[0] == -1 and lst[1] == -1:
            return '5', -1
        
        # If one of the elements is -1, return the other element and its index
        if lst[0] == -1:
            return lst[1], 1
        if lst[1] == -1:
            return lst[0], 0
        
        # If both elements are '5', return '5' and indicate no index
        if lst[0] == '5' and lst[1] == '5':
            return '5', -1
        
        # If one of the elements is '5', return the other element and its index
        if lst[0] == '5':
            return lst[1], 1
        if lst[1] == '5':
            return lst[0], 0

        # Check if one of the elements matches the preferred value, and return it with its index
        # Currently preferred value is based on the ASL-LEX glosses
        # TODO: Find another selection method that is independend of the ASL-LEX glosses, e.g. just choose one at random
        preferred_value = self.handshapes[self.gloss]  # Assuming self.handshapes[self.gloss] is defined elsewhere
        if lst[0] == preferred_value:
            return preferred_value, 0
        if lst[1] == preferred_value:
            return preferred_value, 1
        
        # If '5' is not in the list, make a random choice between the two elements and return it with its index
        chosen_index = random.choice([0, 1])
        return lst[chosen_index], chosen_index



if __name__ == "__main__":
    # Creating an instance of MyClass
    annotatorutils = BaseAnnotationUtils()
    # Read original asllex handshapes to file
    annotatorutils.read_asllex_handshapes()
    