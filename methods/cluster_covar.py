class AutoAnnotate:
    def __init__(self, hand = "L", 
                 path = 'data/HE_raw_data/', 
                 calibration_path = 'data/ASL_calibration.json',
                 semlex_path = 'data/semlex.csv'):
        """
        Class initializer method.
        :param attribute1: Description of attribute1.
        :param attribute2: Description of attribute2.
        """
        self.hand = hand
        self.main_dir = path
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
        
        # Read calibration data
        self.calibration = pd.read_json(calibration_path)
        # Read calibration data
        self.semlex = pd.read_csv(semlex_path)
        self.empty_data_counter = 0
        self.empt_dir = 0
        
        

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

    
    def build_covar(self, df):
        spectrum = []
        for col_name, data in df.items():
            spectrum.append(data.values)
        spectrum = np.array(spectrum).T
        cov_matrix = np.cov(spectrum, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Step 3: Sort the eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        k = 2
        selected_eigenvectors = sorted_eigenvectors[:, :k]
        projected_data = np.dot(spectrum - np.mean(spectrum, axis=0), selected_eigenvectors)
        print(projected_data.shape)
        return projected_data



        #self.semlex['spectrum'] = ...

    def build_database(self, permformer = 'Daniel'):
        pattern = 'P1' + self.hand + '_*.csv'
        pattern_meta = 'P1' + self.hand + '_*Meta.json'
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        spectra_list = []
        glosses = []   

        for subdir in subdirectories:
            
            # Get all files in the current subdirectory
            files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
            data_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern)]
            metadata_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern_meta)]
            try:
                metapath = os.path.join(subdir, metadata_file[0])
                with open(metapath, 'r') as file:
                    lines = file.readlines()  # Read all lines into a list
                    # Access line 7 (note: lists are 0-indexed, so line 7 is at index 6)
                    line_7 = lines[6].strip()
                    match = re.search(performer_pattern, line_7)
                    name = match.group(1)
                    
                    if name == permformer:
                        
                        
                        df, df_timecodes = self.load_sample(subdir, data_file[0])   
                        
                        if df is not None:
                            spectrum = self.build_covar(df)
                            glosses.append(subdir[17:])
                            spectra_list.append(spectrum)
                            
            except: 
                self.empt_dir += 1
                continue
        
        
        print('bugged_files', self.empty_data_counter)
        print('empty_dir', self.empt_dir)
        spectra_df = pd.DataFrame({'label': glosses, 'spectrum': spectra_list})

        return spectra_df
    
    def plot_data(self, df):
        # Unique handshapes and assigning a color to each
        n_cols = 2
        handshapes = df['Handshape'].unique()[0:n_cols]
        print('hs', handshapes)
        print('hs', len(handshapes))
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
                    'peachpuff', 'antiquewhite'][0:n_cols]
        
        color_map = dict(zip(handshapes, colors))
        #keys_to_keep = ['i', 'o', 'curved_5']
        #color_map = {k: color_map[k] for k in keys_to_keep if k in color_map}


        # Plotting
        plt.figure(figsize=(10, 6))
        

        for handshape, color in color_map.items():
            # Filter the DataFrame for each handshape
            subset = df[df['Handshape'] == handshape].reset_index(drop=True)
            print('handshape', len(subset))
            
            # Assume 'spectrum' values are numeric or can be plotted directly
            # This might need to be adjusted based on your 'spectrum' data format
            for i in range(len(subset)):
                try:
                    y = np.arange(0, len(subset.loc[i]['spectrum']))
                    plt.scatter(y, subset.loc[i]['spectrum'], color=color, label=handshape)
                except KeyError:
                    pass
        plt.title('Spectrum Values Color-Coded by Handshape')
        plt.xlabel('Index')  # Adjust if 'spectrum' represents something that has a specific label
        plt.ylabel('Spectrum Value')
        plt.legend(title='Handshape')
        plt.show()
    

    





# Example usage of the class
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module
    import random
    import os
    import fnmatch
    import json 
    import re 

    # Creating an instance of MyClass
    annotator = AutoAnnotate()
    data = annotator.build_database()
    merged_df = pd.merge(data, annotator.semlex, on='label', how='inner')
    annotator.plot_data(merged_df)
    
    #annotator.plot_data(data)

