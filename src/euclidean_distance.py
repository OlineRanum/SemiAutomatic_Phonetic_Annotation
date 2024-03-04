from src.utils.base import BaseAnnotationUtils
from tqdm import tqdm

import os, re, fnmatch, random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



class EuclideanDistance(BaseAnnotationUtils):
    def __init__(self):
        super().__init__()
        
    def annotate(self, hands = ['L', 'R'] ):
        self.read_asllex_handshapes()
        
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        
        handshapes = []
        glosses = []   

        for subdir in tqdm(subdirectories, desc='Annotating Handshapes with Euclidean Distance'):
            # Get gloss name from directory title
            self.gloss = subdir[17:]

            handshape_temp = [] 
            for hand in hands:
                pattern = 'P1'+hand+'_*.csv'
                pattern_meta = 'P1'+hand+'_*Meta.json'
        
                # Find the file matching the pattern, only one such file per subdir
                data_file = next((f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern)), None)
                metadata_file = next((f for f in os.listdir(subdir) if fnmatch.fnmatch(f, pattern_meta)), None)

                try:
                    metapath = os.path.join(subdir, metadata_file)

                    with open(metapath, 'r') as file:
                        df, _ = self.load_sample(subdir, data_file)   
                        
                        if df is not None:
                            # Find the performer name in the metadata file
                            lines = file.readlines()  
                            name = re.search(performer_pattern, lines[6].strip()).group(1)
                            
                            # Read calibration data from performer json file
                            self.calibration = self.read_calibration(subdir + '/Cal_P1'+hand+'_'+name+'.json')    

                            # Calculate Euclidean distance for each time-frame in the dataframe
                            df['closest_calibration'] = df.apply(lambda row: self.ED_estimation(row), axis=1)
                            
                            # Select the new handshape based on frequency
                            new_handshape = self.select_new_handshape(df)
                            handshape_temp.append(new_handshape)  

                        else:
                            # If file not available, i.e. only handshape '5' was recorded, append -1
                            handshape_temp.append(-1) 

                except: 
                    self.empt_dir += 1
                    handshape_temp.append(-1)
                    continue
                
            hs, _ = self.select_hand(handshape_temp)
            handshapes.append(hs)
            glosses.append(self.gloss)

        print('empty_files: ', self.empty_data_counter)
        print('empty_directories: ', self.empt_dir)
        self.output_txt(glosses, handshapes)

    def ED_estimation(self, time_series_row, cal = None):
        """ Calculate the Euclidean distance between the hand pose frames and the calibration poses.	
        """
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
             
    def plot_ED_heatmap(self, gloss):
            """ Plot the time-series labeling of the handshapes for a given gloss
            """
            from matplotlib.patches import Patch
            from matplotlib.ticker import MaxNLocator, FormatStrFormatter
            
            df, dt = self.load_sample('data/HE_raw_data/'+gloss, '/P1L_Daniel.csv')
            cal = self.read_calibration('data/HE_raw_data/'+gloss+'/Cal_P1L_Daniel.json')
            df['closest_calibration'] = df.apply(lambda row: self.ED_estimation(row, cal=cal), axis=1)
           
           # Set time-axis
            dt['Timecode(master)'] = dt['Timecode(master)'].astype(str).str.replace('(.*):(\\d{3})$', '\\1.\\2', regex=True)
            dt['Timecode(master)'] = pd.to_timedelta(dt['Timecode(master)'])
            time_deltas = dt['Timecode(master)'] - dt['Timecode(master)'].iloc[0]
            df['time_s'] = time_deltas.dt.total_seconds()

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
            
            plt.savefig('figures/leastsquares_annotations.png', bbox_inches='tight')



        


# Example usage of the class
if __name__ == "__main__":
    # Creating an instance of MyClass
    annotator = EuclideanDistance()
   
    """
    annotator.get_original_handshapes()
    glosses, handshapes = annotator.build_database()
    annotator.output_txt(glosses, handshapes)
    """

    gloss = 'zero'
    annotator.plot_ED_heatmap(gloss)
