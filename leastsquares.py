from utils.base import AutoAnnotate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import fnmatch
import json 
import re 

class LS(AutoAnnotate):
    def __init__(self):
        # Call the parent class's constructor using super()
    
        super().__init__()

    

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
        with open('output/wlasl_989.json', 'r') as file:
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
  

    def find_closest_calibration(self, time_series_row, cal = None):
        if cal is not None:
            self.calibration = cal
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
        handshapes = []
        glosses = []   


        for subdir in subdirectories:
            self.gloss = subdir[17:]
            handshape_temp = [] 

            for hand in hands:
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
                except: 
                    self.empt_dir += 1
                    handshape_temp.append(-1)
                    continue
                
            hs, _ = self.choose_element(handshape_temp)
            handshapes.append(hs)
            glosses.append(self.gloss)

            print(hs)
                    
        


        print('bugged_files', self.empty_data_counter)
        print('empty_dir', self.empt_dir)
        return glosses, handshapes
            
    def choose_element(self, lst):
        print(lst)
        # Check if the list contains exactly two elements
        if len(lst) != 2:
            return None, None  # Return None for both value and index
        
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
        
        preferred_value = self.handshapes[self.gloss]  # Assuming self.handshapes[self.gloss] is defined elsewhere
        if lst[0] == preferred_value:
            return preferred_value, 0
        if lst[1] == preferred_value:
            return preferred_value, 1
        
        # If '5' is not in the list, make a random choice between the two elements and return it with its index
        chosen_index = random.choice([0, 1])
        return lst[chosen_index], chosen_index


    
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
        filtered_proportions = proportions[proportions >= 0.05]

        # Special handling for '5', which is also the baseline noise value in the dataset
        special_values = ['5']
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
        
    def scatter_plot(self, df):
        plt.figure(figsize=(16,10))
        scatter = plt.scatter(df['tsne-2d-one'], df['tsne-2d-two'], c=df['new_labels'], cmap='plasma', alpha=1)

        plt.title('t-SNE projection of the dataset')
        plt.xlabel('t-SNE axis 1')
        plt.ylabel('t-SNE axis 2')
        
        # Optional: add a legend
        # Correct way to add a legend based on scatter
        legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.gca().add_artist(legend1)  # This is how you could use add_artist, but it's on an axes object

        
        plt.show()
    

    def scatter_plot(self, df):
        # Assuming 'closest_calibration' is a column in df containing the pose names
        sns.set_theme(style="dark")
        # Map unique pose names to colors
        unique_poses = df['closest_calibration'].unique()
        color_map = {pose: color for pose, color in zip(unique_poses, plt.cm.tab20(np.linspace(0, 1, len(unique_poses))))}
        
        # Create a scatter plot
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Assign a color to each row based on its closest_calibration value
        it = 0
        
        for pose, color in color_map.items():
                
            times = df[df['closest_calibration'] == pose]['time_s'].astype(float).tolist()

            y_values = [it] * len(times)  # Use pose as y-value for differentiation
            ax.scatter(times, y_values, label=pose, color=color, alpha=1, edgecolors='w', s=100)
            it += 0.2

        
        # Since the y-axis doesn't represent anything specific, hide it for clarity
        ax.yaxis.set_visible(False)
        
        # Add a legend below the plot horizontally
        #ax.legend(title="Pose Name", bbox_to_anchor=(0.5, -0.2), loc='bottom center', ncol=len(unique_poses))
    
        plt.title("Least Square Pose Approximation")
        plt.xlabel("Time [s]")
        plt.ylim(-0.2, 0.8)
        
        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.7])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
                fancybox=True, shadow=True, ncol=len(unique_poses))
        #plt.tight_layout(rect=[0, 0.15, 1, 1])

        
        plt.savefig('figure.png', bbox_inches='tight')

    def heatmap(self, df):
        from matplotlib.patches import Patch
        from matplotlib.ticker import MaxNLocator, FormatStrFormatter
        sns.set_theme(style="white")
        df['CategoryCode'], uniques = pd.factorize(df['closest_calibration'])

        colors = [
            (0.445163, 0.322724, 0.506901, 1.0),  # Pink
            (0.944006, 0.377643, 0.365136, 1.0),  # Purple
            (0.001462, 0.000466, 0.013866, 1.0),  # Black
            (0.987053, 0.991438, 0.749504, 1.0)   # White
        ]

        color_map = {code: colors[i] for i, code in enumerate(df['CategoryCode'].unique())}
        bar_colors = df['CategoryCode'].map(color_map)
        
        fig, axs = plt.subplots(2, 1, figsize=(4.3, 1.55))
        plt.subplots_adjust(hspace=3)  # Adjust the vertical spacing between subplots
    
        # Original Range
        x = df['time_s'].astype(float)
        axs[0].bar(x, 0.1 * np.ones(len(df)), width = 0.4, color=bar_colors, edgecolor='none')
        axs[0].yaxis.set_visible(False)
        axs[0].xaxis.set_major_locator(MaxNLocator(integer=True)) 
        axs[0].set_xlim(df['time_s'].min(), df['time_s'].max() + 0.1)
        #axs[0].set_title("a) Full Recording", loc='left', fontsize = 11)  # Set title for the first subplot
        axs[0].tick_params(axis='x', labelsize='small')

        
        # Zoomed-In Range
        bar_width_zoomed = 0.0005  # Adjust as needed
        axs[1].bar(x, 0.1 * np.ones(len(df)), width=bar_width_zoomed, color=bar_colors, edgecolor='none')
        axs[1].set_xlim(0.989, 1.016)  # Adjust as needed for zoomed-in view
        axs[1].yaxis.set_visible(False)
        axs[1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # Format x-ticks with two decimals
        axs[1].set_title("", loc='left', fontsize = 11)  # Set title for the second subplot
        axs[1].set_xticks([0.99, 1.00, 1.01])
        axs[1].tick_params(axis='x', labelsize='small')

        for ax in axs:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
       
        
        # Shared elements
        legend_elements = [Patch(facecolor=color, edgecolor=color, label=label) for color, label in zip(colors, uniques)]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.16),
                fancybox=True, ncol=4, frameon=False)
        fig.suptitle("Temporal Segmentation of LS Poses\n", fontsize=13, y=1.27)
        axs[-1].set_xlabel('Time [s]', fontsize=11, labelpad=10)

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
        
        plt.savefig('figure.png', bbox_inches='tight')

        


# Example usage of the class
if __name__ == "__main__":
    # Creating an instance of MyClass
    annotator = LS()
    
    gloss = 'zero'
    df, dt = annotator.load_sample('data/HE_raw_data/'+gloss, '/P1L_Daniel.csv')
    cal = annotator.read_calibration('data/HE_raw_data/'+gloss+'/Cal_P1L_Daniel.json')
    df['closest_calibration'] = df.apply(lambda row: annotator.find_closest_calibration(row, cal=cal), axis=1)
    # Ensure the column is treated as a string and replace ':' with '.' before the milliseconds
    dt['Timecode(master)'] = dt['Timecode(master)'].astype(str).str.replace('(.*):(\\d{3})$', '\\1.\\2', regex=True)

    # Convert 'Timecode(master)' to timedelta
    dt['Timecode(master)'] = pd.to_timedelta(dt['Timecode(master)'])

    # Subtract the first timestamp to get the duration from the start
    time_deltas = dt['Timecode(master)'] - dt['Timecode(master)'].iloc[0]

    # Convert timedeltas to seconds
    time_seconds = time_deltas.dt.total_seconds()

    df['time_s'] = time_seconds

    annotator.heatmap(df)
    """
    annotator.get_original_handshapes()
    glosses, handshapes = annotator.build_database()
    annotator.write_glosses_and_handshapes_to_file(glosses, handshapes)
    """