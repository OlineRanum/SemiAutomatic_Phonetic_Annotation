from utils.base import AutoAnnotate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import ast
import os
import seaborn as sns
            # Perform t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
import fnmatch
import json 
import re 
from leastsquares import LS
from sklearn.cluster import KMeans

class KMEANS(LS):
    def __init__(self):
        # Call the parent class's constructor using super()
        
        super().__init__()
        self.empt_dir = []
        self.empty_data_counter = 0
        self.empty_data_counter_both = 0
    
    def write_dict_to_file(self, avg_dict, filename='gloss_handshape_dict.txt'):
        """
        Writes glosses and handshapes to a file, with each pair on a new line.

        :param glosses: List of glosses.
        :param handshapes: List of handshapes.
        :param filename: Name of the file to write to.
        """
        with open(filename, 'w') as file:
            for gloss, data in avg_dict.items():
                file.write(f"{gloss}, {data}\n")
        
        print(f"Data has been written to {filename}.")
    
    def read_from_file(self, filename='gloss_handshape_dict.txt'):
        data = []
        with open(filename, 'r') as file:
            for line in file:
                # Split the line into gloss and the dictionary part
                gloss, dict_str = line.split(',', 1)
                gloss = gloss.strip()
                # Convert the dictionary string to an actual dictionary
                values = ast.literal_eval(dict_str.strip())
                # Extract the values you need, assuming 'Handshape' and 'ls_id' keys exist
                row = {'gloss': gloss, 'mean_handshape': values['Handshape'], 'hs': values['ls_id']}
                data.append(row)
        
        # Convert the list of dictionaries into a pandas DataFrame
        df = pd.DataFrame(data)
        return df

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


        
    def build_database(self, permformer = 'Daniel'):
        hands = ['L', 'R']  
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        avg_dict = {} 

        for subdir in subdirectories:
            self.gloss = subdir[17:]
            handshape_temp = [] 
            
            for hand in hands:
                dual = 0
                pattern = 'P1'+hand+'_*.csv'
                pattern_meta = 'P1'+hand+'_*Meta.json'
        
            
                # Get all files in the current subdirectory
                files = [f for f in os.listdir(subdir) if os.path.isfile(os.path.join(subdir, f))]
                data_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern)]
                metadata_file = [f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern_meta)]
                
            
                try:
                    
                    metapath = os.path.join(subdir, metadata_file[0])
                    

                    with open(metapath, 'r') as file:
                        
                    
                        df, df_timecodes = self.load_sample(subdir, data_file[0])
                        if hand == 'L':
                            df_L = df   
                        

                        if df is not None:
                            
                            lines = file.readlines()  # Read all lines into a list
                            # Access line 7 (note: lists are 0-indexed, so line 7 is at index 6)
                            line_7 = lines[6].strip()
                            match = re.search(performer_pattern, line_7)
                            
                            name = match.group(1)
                            self.calibration = self.read_calibration(subdir + '/Cal_P1'+hand+'_'+name+'.json')    
                            
                            df['closest_calibration'] = df.apply(lambda row: self.find_closest_calibration(row), axis=1)
                            
                            new_handshape = self.get_handshape(df)

                            handshape_temp.append(new_handshape)
                            

                        else:
                            handshape_temp.append(-1)
                            self.empty_data_counter += 1
                            dual += 1
                            if dual == 2:
                                self.empty_data_counter_both += 1 

                except: 
                    self.empt_dir.append(subdir)
                    handshape_temp.append(-1)

                    exit()
                
            hs, idx = self.choose_element(handshape_temp)
            if idx == 0:
                df = df_L
            
            
            try:
                df = df[df['closest_calibration'] == hs]
                df = df.drop('closest_calibration', axis=1)
                
                
                mean = df.mean().tolist()
                avg_dict[self.gloss] = {'Handshape': mean, 'ls_id': hs}
            except KeyError:
                pass       

        


        print('bugged_files', self.empty_data_counter)
        print('dual_bugged_files', self.empty_data_counter_both)
        print('empty_dir', self.empt_dir)
        
        return avg_dict
    
    def k_means(self, df):
        k = 50
        kmeans = KMeans(n_clusters=k)
        
        data = df['mean_handshape'].tolist()
        
        kmeans.fit(data)

        labels = kmeans.labels_

        df['new_labels'] = labels

        sns.set_theme(style="darkgrid")
        sns.despine(bottom = False, left = True)
        sns.set(rc={"xtick.bottom" : True,"xtick.top" : True, "ytick.left" : True, "ytick.right" : True})

        
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(np.array(data))

        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]

        # Plotting
        plt.figure(figsize=(8,6))
        
        plt.scatter(df['tsne-2d-one'], df['tsne-2d-two'], c=df['new_labels'], cmap='plasma', alpha=0.6)
        
        ax = plt.gca()  # Get current axes
        spine_width = 2  # Adjust the width of the border
        spine_color = 'black'  # Adjust the color of the border
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
            spine.set_edgecolor(spine_color)
        plt.title('t-SNE Projection of Handshapes\n', size=26)

        # Attempt to re-enable ticks explicitly for visibility
        ax.tick_params(axis='x', which='both', length=6, width=2, labelbottom=False)  # Adjust 'length' and 'width' as needed
        ax.tick_params(axis='y', which='both', length=6, width=2, labelleft=False)  # Adjust 'length' and 'width' as needed

        plt.xlabel('')
        plt.ylabel('')

        plt.savefig('kmeans.png', bbox_inches='tight')

        return df
        
# Example usage of the class
if __name__ == "__main__":
    # Creating an instance of MyClass
    annotator = KMEANS()
    #annotator.get_original_handshapes()
    #avg_dict = annotator.build_database()
    #annotator.write_dict_to_file(avg_dict)
    data = annotator.read_from_file()
    df = annotator.k_means(data)
    annotator.write_glosses_and_handshapes_to_file(df['gloss'], df['new_labels'], filename='gloss_handshape_pairs_kmeans.txt')
