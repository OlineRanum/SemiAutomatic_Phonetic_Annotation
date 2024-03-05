from src.euclidean_distance import EuclideanDistance

import os, fnmatch, re
from tqdm import tqdm 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class K_Means(EuclideanDistance):
    def __init__(self):
        super().__init__()

    def annotate(self, hands = ['L', 'R']):
        avg_dict = {} 
        self.read_asllex_handshapes()
        
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        

        for subdir in tqdm(subdirectories, desc='Annotating Handshapes with Euclidean Distance'):
            # Get gloss name from directory title
            self.gloss = subdir[17:]

            handshape_temp = []
            hand_counter = 0 
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

                        if hand == 'L':
                            df_L = df   

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
                            self.empty_data_counter += 1
                            hand_counter += 1 
                            if hand_counter == 2:
                                self.empty_data_counter -= 2
                                self.empty_data_counter_both_hands += 1  
                except: 
                    handshape_temp.append(-1)
                    self.empty_data_counter += 1
                    hand_counter += 1 
                    if hand_counter == 2:
                        self.empty_data_counter -= 2
                        self.empty_data_counter_both_hands += 1  
                    
                
            hs, idx = self.select_hand(handshape_temp)

            if idx == 0:
                df = df_L
            
            
            try:
                df = df[df['closest_calibration'] == hs].drop('closest_calibration', axis=1)
                mean = df.mean().tolist()
                if not np.isnan(mean).any():
                    avg_dict[self.gloss] = {'Handshape': mean, 'ls_id': hs}

            except (KeyError, TypeError):
                print('Data Error, check directory:',   subdir)
                pass       

        
        print('Occurances where one hand recorded no data: ', self.empty_data_counter)
        print('Occurances where both hands recorded no data: ', self.empty_data_counter_both_hands)

        
        self.write_dict_to_file(avg_dict, filename='output/KMeans_dict.txt')


    def KMeans_estimation(self, df, k =4):
        """ Calculate K-Means clustering for the handshapes to set new labels for the data.
        """
        kmeans = KMeans(n_clusters=k)
        
        data = df['mean_handshape'].tolist()

        kmeans.fit(data)
        labels = kmeans.labels_
        df['new_labels'] = labels
        return df

    def plot_KMeans_scatter(self, df): 
        sns.set_theme(style="darkgrid")
        sns.despine(bottom = False, left = True)
        sns.set(rc={"xtick.bottom" : True,"xtick.top" : True, "ytick.left" : True, "ytick.right" : True})

        
        tsne = TSNE(n_components=2, random_state=42)
        
        data = df['mean_handshape'].tolist()
        tsne_results = tsne.fit_transform(np.array(data))

        df['tsne-2d-one'] = tsne_results[:,0]
        df['tsne-2d-two'] = tsne_results[:,1]

        plt.figure(figsize=(8,6))
        plt.scatter(df['tsne-2d-one'], df['tsne-2d-two'], c=df['new_labels'], cmap='plasma', alpha=0.6)
        
        ax = plt.gca()  
        spine_width = 2  
        spine_color = 'black' 
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)
            spine.set_edgecolor(spine_color)

        plt.title('t-SNE Projection of Handshapes\n', size=26)
        ax.tick_params(axis='x', which='both', length=6, width=2, labelbottom=False)  # Adjust 'length' and 'width' as needed
        ax.tick_params(axis='y', which='both', length=6, width=2, labelleft=False)  # Adjust 'length' and 'width' as needed

        plt.xlabel('')
        plt.ylabel('')

        plt.savefig('figures/kmeans.png', bbox_inches='tight')
        
# Example usage of the class
if __name__ == "__main__":
    # Creating an instance of MyClass
    annotator = K_Means()

    annotator.annotate()    
    data = annotator.read_KMeans_dict(filename = 'output/KMeans_dict.txt')
    df = annotator.KMeans_estimation(data)
    annotator.plot_KMeans_scatter(df)
    annotator.output_txt(df['gloss'], df['new_labels'], filename='output/kmeans_labels.txt')
    