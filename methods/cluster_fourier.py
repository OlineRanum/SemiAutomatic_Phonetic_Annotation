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
        self.semlex['spectrum'] = 0
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
    
    def fourier_transform(self, signal):
            signal_fft = np.fft.fft(signal)
            n = len(signal)  # Length of the signal
            fs = 100
            frequencies = np.fft.fftfreq(n, d=1/fs)
            return signal_fft, frequencies
    
    def plot_signal(self, signal_fft, frequencies, n):
        #signal_fft, frequencies = self.fourier_transform(signal)

        # Only plot the first half of frequencies, as the second half is the mirror image
        half_n = int(n) // 2

        # Plotting the magnitude spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies[:half_n], np.abs(signal_fft)[:half_n] * 2 / n)  # Normalize amplitude
        plt.title('Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude |X(f)|')
        plt.grid(True)
        plt.show()



    def build_spectra(self, df):
        spectrum = np.zeros(len(df), dtype='complex128')
        for col_name, data in df.items():
            signal_fft, frequencies = self.fourier_transform(data.values)
            spectrum += signal_fft
        # Normalize by number of frames
        spectrum /= len(df)
        return spectrum, frequencies


        #self.semlex['spectrum'] = ...

    def build_database(self, permformer = 'Daniel'):
        pattern = 'P1' + self.hand + '_*.csv'
        pattern_meta = 'P1' + self.hand + '_*Meta.json'
        performer_pattern = r'"performerName":\s*"(.+?)"'
        subdirectories = [os.path.join(self.main_dir, d) for d in os.listdir(self.main_dir) if os.path.isdir(os.path.join(self.main_dir, d))]
        spectra_list = []
        freq_list = []
        glosses = []   
        n_list = [] 
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
                            spectrum, frequencies = self.build_spectra(df)
                            n_list.append(len(df))
                            glosses.append(subdir[17:])
                            spectra_list.append(spectrum)
                            freq_list.append(frequencies)
            except: 
                self.empt_dir += 1
                continue
        
        print('bugged_files', self.empty_data_counter)
        print('empty_dir', self.empt_dir)

        return spectra_list, freq_list, glosses, n_list
    
    def writeto_csv(self, data, path):
        spectra_list, freq_list, glosses = data
        print(len(spectra_list), len(freq_list), len(glosses))
        df = pd.DataFrame({'gloss': glosses, 'spectrum': spectra_list, 'frequencies': freq_list})

        df.to_csv(path, index=False)
        print('Data saved to', path)
    
    def plot_data(self, data):
        spectra_list, freq_list, glosses, n_list = data
        # Plotting the magnitude spectrum
        plt.figure(figsize=(10, 6))
        plt.title('Magnitude Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude |X(f)|')
        plt.grid(True)
        for i in range(len(spectra_list)):
            self.plot_signal2(freq_list[i], spectra_list[i], n_list[i])
        plt.show()

    def plot_signal2(self,frequencies, signal_fft, n ):

        # Only plot the first half of frequencies, as the second half is the mirror image
        half_n = int(n) // 2

        plt.plot(frequencies[:half_n], np.abs(signal_fft)[:half_n] * 2 / n)  # Normalize amplitude




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
    annotator.plot_data(data)