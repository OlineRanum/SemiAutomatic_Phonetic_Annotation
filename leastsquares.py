from utils.base import AutoAnnotate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import fnmatch
import json 
import re 

class LS(AutoAnnotate):
    def __init__(self):
        # Call the parent class's constructor using super()
        
        gloss = 'hat'
        hand = 'R'
        super().__init__(calibration_path = 'data/HE_raw_data/'+gloss+'/Cal_P1'+hand+'_Daniel.json')
        self.get_original_handshapes()
        glosses, handshapes = self.build_database()
        self.write_glosses_and_handshapes_to_file(glosses, handshapes)
    
        #file = 'WUgXGr1GImMWVYNi23V6'
        
        #df, timecodes = self.load_sample('data/HE_raw_data/' + gloss, 'P1'+hand+'_Daniel.csv')
        #df['closest_calibration'] =   df.apply(lambda row: self.find_closest_calibration(row), axis=1)

        #print(df['closest_calibration'])
        #print(self.calibration.head())
        #self.scatter_plot(df)
    def write_glosses_and_handshapes_to_file(self, glosses, handshapes, filename='gloss_handshape_pairs.txt'):
        """
        Writes glosses and handshapes to a file, with each pair on a new line.

        :param glosses: List of glosses.
        :param handshapes: List of handshapes.
        :param filename: Name of the file to write to.
        """
        if len(glosses) != len(handshapes):
            print("The lists glosses and handshapes must be of the same length.")
            return
        
        with open(filename, 'w') as file:
            for gloss, handshape in zip(glosses, handshapes):
                file.write(f"{gloss}, {handshape}\n")
        
        print(f"Data has been written to {filename}.")
    
    def get_original_handshapes(self):
        with open('wlasl_800.json', 'r') as file:
            data = json.load(file)

        # Step 2 & 3: Process the data and save to a dictionary
        gloss_handshape_dict = {}
        for entry in data:
            gloss = entry['gloss']
            # Ensure there is at least one instance and it has the 'Handshape' key
            if entry['instances'] and 'Handshape' in entry['instances'][0]:
                handshape = entry['instances'][0]['Handshape']
                gloss_handshape_dict[gloss] = handshape

        # Step 4: Write to a new JSON file
        with open('gloss_handshape.json', 'w') as outfile:
            json.dump(gloss_handshape_dict, outfile, indent=4)

        print("Data has been processed and saved.")
        self.handshapes = gloss_handshape_dict
        print(self.handshapes)

    def find_closest_calibration(self, time_series_row):
        # Convert the series row to a numpy array for distance computation
        ts_row_array = time_series_row.to_numpy()
        
        # Initialize a variable to keep track of the minimum distance and the name of the closest pose
        min_distance = np.inf
        closest_pose = None
        
        # Iterate over each row in the calibration DataFrame to calculate distances
        for pose_name, row in self.calibration.iterrows():
            joints = np.array(row['Joints'])
            distance = np.linalg.norm(ts_row_array - joints)
            
            if distance < min_distance:
                min_distance = distance
                closest_pose = pose_name
        
        return closest_pose 

    def scatter_plot(self, df):
        # Assuming 'closest_calibration' is a column in df containing the pose names
        
        # Map unique pose names to colors
        unique_poses = df['closest_calibration'].unique()
        color_map = {pose: color for pose, color in zip(unique_poses, plt.cm.tab20(np.linspace(0, 1, len(unique_poses))))}
        
        # Create a scatter plot
        fig, ax = plt.subplots()
        
        # Assign a color to each row based on its closest_calibration value
        for pose, color in color_map.items():
            indices = df[df['closest_calibration'] == pose].index
            ax.scatter(indices, [1] * len(indices), label=pose, color=color, alpha=0.6, edgecolors='w')
        
        # Since the y-axis doesn't represent anything specific, hide it for clarity
        ax.yaxis.set_visible(False)
        
        # Add a legend outside the plot to the right
        ax.legend(title="Pose Name", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title("Closest Calibration Pose Over Time")
        plt.xlabel("Time Point Index")
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        
        plt.show()
        
    def build_database(self, permformer = 'Daniel'):
        pattern = 'P1' + self.hand + '_*.csv'
        pattern_meta = 'P1' + self.hand + '_*Meta.json'
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        handshapes = []
        glosses = []   

        for subdir in subdirectories:
            
            # Get all files in the current subdirectory
            files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
            data_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern)]
            metadata_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern_meta)]
            try:
                metapath = os.path.join(subdir, metadata_file[0])

                with open(metapath, 'r') as file:
                
                    df, df_timecodes = self.load_sample(subdir, data_file[0])   

                    if df is not None:
                        self.gloss = subdir[17:]
                        lines = file.readlines()  # Read all lines into a list
                        # Access line 7 (note: lists are 0-indexed, so line 7 is at index 6)
                        line_7 = lines[6].strip()
                        match = re.search(performer_pattern, line_7)
                        name = match.group(1)
                        self.calibration = self.read_calibration(subdir + '/Cal_P1'+self.hand+'_'+name+'.json')    
                        df['closest_calibration'] = df.apply(lambda row: self.find_closest_calibration(row), axis=1)
                        new_handshape = self.get_handshape(df)
                            
                        glosses.append(self.gloss)
                        handshapes.append(new_handshape)
                        print(new_handshape)

                    else:
                        handshapes.append(-1) 
          


            except: 
                self.empt_dir += 1
                glosses.append(subdir[17:])
                handshapes.append('Reach')
                continue
        return glosses, handshapes
            
        print('bugged_files', self.empty_data_counter)
        print('empty_dir', self.empt_dir)
    
    def get_handshape(self, df):
        print(self.gloss, self.handshapes[self.gloss])
        
        value_counts = df['closest_calibration'].value_counts(dropna=False)
        return self.calculate_value_proportions(value_counts)
        #print(value_counts)
        #print('max:')
        #print(self.calculate_value_proportions(value_counts))
        #print('--------------------')
    
    def calculate_value_proportions(self, value_counts):
        """
        This function takes a pandas Series (or DataFrame column) as input,
        calculates the value counts, and then computes the proportion of each
        unique value relative to the total count.

        :param column: pandas Series
        :return: pandas Series containing the proportions of each unique value
        """
        total_count = value_counts.sum()
        proportions = value_counts / total_count
        filtered_proportions = proportions[proportions >= 0.1]

                # Special handling for 'Paddle' and 'Reach'
        special_values = ['Paddle', 'Reach']
        # Check if special values are present and filter accordingly
        filtered_specials = filtered_proportions[filtered_proportions.index.isin(special_values)]
        
        if not filtered_specials.empty and len(filtered_proportions) > len(filtered_specials):
            # If there are special values and other values, exclude special values
            return filtered_proportions[~filtered_proportions.index.isin(special_values)].idxmax()
        elif not filtered_specials.empty:
            # If only special values are present, keep the one with the highest ratio
            return filtered_specials.idxmax()
        else:
            # Return the value with the highest proportion if no special values are present
            return filtered_proportions.idxmax()
        


    


        


# Example usage of the class
if __name__ == "__main__":
    # Creating an instance of MyClass
    annotator = LS()
