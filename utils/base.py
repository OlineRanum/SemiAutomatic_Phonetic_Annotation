import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import fnmatch
import json 
import re 

class AutoAnnotate:
    def __init__(self, hand = "R", 
                 path = 'data/HE_raw_data/'):
        """
        Class initializer method.
        :param attribute1: Description of attribute1.
        :param attribute2: Description of attribute2.
        """

        self.hand = hand
        self.main_dir = path
        
        # Selected columns from the CSV file
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

        with open(file_path, 'r') as file:
            data = json.load(file)
        poses_data = []
        

        for pose_id, pose_info in data['calibration'][0]['poses'].items():
            pose_name = pose_info['name']
            joints = pose_info['joints'][3:]
            # Append a tuple of (pose_name, joints) to the list
            poses_data.append((pose_name, joints))

        # Create a DataFrame
        df = pd.DataFrame(poses_data, columns=['Pose Name', 'Joints'])
        
        df.set_index('Pose Name', inplace=True)
        droprows = ['Fist', 'Paddle', 'Reach', 'Fist Thumb Out', 'Thumbs Up', 'open_b_tight']
        for row in droprows:
            if row in df.index:
                df = df.drop(index=row)

        
        
        return df

    
            
    def get_handshape(self, df):

        value_counts = df['closest_calibration'].value_counts(dropna=False)
        return self.calculate_value_proportions(value_counts)
    


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
        
    
    def load_sample(self, subdir, file):
        
        try:
            df = pd.read_csv(subdir +'/'+ file)
            df.columns = df.columns.str.replace(' ', '')
            df_timecodes = df[self.selected_cols[0:3]]
            df = df[self.selected_cols[6:]]
            return df, df_timecodes
        
        except Exception as e:
            self.empty_data_counter += 1
            
            return None, None

    
    def plot_handshape_timeseries(self, df, cal):
        # Plot time evolution of a single handshape 
        plt.figure(figsize=(10, 6))
        colors = ['mistyrose',  'orchid',
                    'blueviolet', 'lemonchiffon',
                    'skyblue', 'salmon',
                    'olive', 'lightskyblue',
                    'royalblue', 'mediumseagreen',
                    'greenyellow', 'silver',
                    'darkslateblue', 'lightgreen',
                    'springgreen', 'darkgray',
                    'lavenderblush', 'darkslategrey',
                    'lightslategrey', 'darkgreen',
                    'cornsilk', 'pink', 'palegoldenrod',
                    'mediumaquamarine', 'dimgray',
                    'indigo', 'lightgray',
                    'burlywood', 'firebrick',
                    'tan', 'deeppink', 'lightslategray',
                    'lightyellow', 'darkgrey',
                    'lightgoldenrodyellow',
                    'ivory', 'grey',
                    'paleturquoise',
                    'white', 'darksalmon',
                    'black', 'turquoise',
                    'peachpuff', 'antiquewhite']

        for i, column in enumerate(df.columns):
            try:
                plt.plot(df.index, df[column], color=colors[i], label=column)
            except:
                pass    
        for i, column in enumerate(df.columns):
            try:
                a = self.calibration.loc['1']['Joints']
            except AttributeError:
                self.calibration = cal
            try:
                plt.plot(df.index, np.full(len(df.index),self.calibration.loc['1']['Joints'][i]),'--', color=colors[i],)
            except:
                pass

        plt.title('Time evolution of measurements')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=round(len(df.columns)/10))

        plt.show()



# Example usage of the class
if __name__ == "__main__":


    # Creating an instance of MyClass
    annotator = AutoAnnotate()
    df, _ = annotator.load_sample('data/HE_raw_data/add', '/P1L_Daniel.csv')
    cal = annotator.read_calibration('data/HE_raw_data/add/Cal_P1L_Daniel.json')
    annotator.plot_handshape_timeseries(df, cal)
    annotator.scatter_plot(df)
