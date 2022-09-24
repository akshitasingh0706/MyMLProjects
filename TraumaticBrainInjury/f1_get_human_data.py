import os
from tracemalloc import start
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import mne

from f0_participants_info import *

cd = "/content/drive/My Drive/UCIClasses/DeepLearning/Project" # path to project
root_path = '/content/drive/My Drive/UCIClasses/DeepLearning/Project'

class get_human_data:
    def __init__(self,
        data_folder: str = root_path + '/Human/data',
        data_label: list = ['data1','data2'],
        control_data1: list = ['102','208'],
        tbi_data1: list = ['244','340'],
        control_data2: list = ['XVZ2FYATE8M0SSF','XVZ2FYATE8X4YXQ'],
        tbi_data2: list = ['XVZ2FYAQH8WVIUC','XVZ2FYATE84MSWI'],    
        fsh_data1 = 200, 
        fsh_data2 = 200, 
        
        freq_band: str = 'normal',   # 'delta','theta','alpha','sigma','beta','gama','normal'
        ica: bool = True,
        filter_domain: str = ['time', 'freq'],
        
        sleep_stage: str = 'N2', 
        channel: list = ['F3', 'F4', 'C3', 'C4', 'O1','O2'],
        epoch_len: int = 30,
        features: list = ['absolute_power','relative_power','slow_fast','frequency amplitude asymmetry',
                        'phase synchrony','coherence','hjorth','spectral_entropy','phase amplitude coupling']
        ):
        self.data_folder = data_folder
        self.data_label = data_label
        self.control_data1 = control_data1
        self.tbi_data1 = tbi_data1
        self.control_data2 = control_data2
        self.tbi_data2 = tbi_data2
        self.fsh_data1 = fsh_data1
        self.fsh_data2 = fsh_data2

        self.freq_band = freq_band
        self.ica = ica
        self.filter_domain = filter_domain
        self.time_params = [0.5, 50] if filter_domain == 'time' else None

        self.sleep_stage = sleep_stage
        self.channel = channel
        self.epoch_len = epoch_len
        self.features = features
    
    def organize_data(self, data_label, class_label): # dataset_label: (dataset1, dataset2), class_label: (control, tbi)
        if data_label == "data1":
            if class_label == "control":
                return self.control_data1
            elif class_label == "tbi":
                return self.tbi_data1
            else:
                print("Incorrect class label")
        elif data_label == "data2":
            if class_label == "control":
                return self.control_data2
            elif class_label == "tbi":
                return self.tbi_data2
            else:
                print("Incorrect class label")
        else:
            print("Incorrect data label")
    
    def run(self, data_label, class_label): 
        data_dict = {}
        sleep_label={}
        subjects = self.organize_data( data_label, class_label)

        for subject in subjects:
            # 1) Load staging files (required for future preproccing steps in raw data)
            if data_label == 'data2':
                stage_data = pd.read_csv(self.data_folder + '/' + subject + '_Stage.txt' ,header=None, index_col=None)
            elif data_label == 'data1':
                temp_df = pd.read_csv(self.data_folder + '/' + subject + '_Stage.txt', header=None, index_col=None)
                temp_df = temp_df.drop(temp_df.index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
                temp_loc  = temp_df.loc[16]
                stage_data  = temp_loc.str.split(expand=True)

                for j in range(17,len(temp_df)+16):
                    temp_loc = temp_df.loc[j]
                    temp = temp_loc.str.split(expand=True)
                    stage_data = stage_data.append(temp)

            stage_data.columns = stage_data.iloc[0]
            stage_data = stage_data[stage_data.iloc[:,20] != 'Stg']

            # 2) Load raw EEG data from the .fif and/or .edf files
            if self.ica:
                raw_data = mne.io.read_raw_fif(self.data_folder + '/' + subject + "_ica.fif", preload=True,verbose=None)
            elif not self.ica:
                raw_data = mne.io.read_raw_edf(self.data_folder + '/' + subject +".edf", preload=True,verbose=None)

            # Filter the data based on the time domain (if specified)
            if self.filter_domain == 'time':
                filtered_data = raw_data.copy()
                filtered_data.load_data().filter(self.time_params[0], self.time_params[1])    
                raw_data = filtered_data

            # 3) Process the raw data
            # 3.1) Get actual signal values from the raw data
            raw_eeg = raw_data[:, :][0]

            # 3.2) Specify the channels we want our data to be limited to
            temp = set(self.channel)
            channel_index = [i for i, val in enumerate(raw_data.ch_names) if val in temp]
            raw_data = (raw_eeg[channel_index,:])

            data = np.zeros((len(self.channel),1))
            stage_temp = []
            for i in range(len(stage_data)):
                if stage_data.iloc[i,20] in self.sleep_stage:
                    stage_temp.append(stage_data.iloc[i,20])
                    fsh_val = self.fsh_data2 if data_label == 'data2' else self.fsh_data1
                    start = i * fsh_val * self.epoch_len
                    stop = i * fsh_val*self.epoch_len + (fsh_val * self.epoch_len)
                    if data_label == 'data2':
                        data = np.concatenate((data, (raw_data[:,start:stop])),axis=1)   
                    elif data_label == 'data1':
                        data = np.concatenate((data, (raw_data[:,start:stop])*10**6),axis=1)   

            data = data[:,1:]
            if np.shape(data)[1] !=0:
                data = data - data.mean(axis=1, keepdims=True)
                data_dict[subject] = data
                if class_label == 'Control':
                    temp = [x + '0' for x in stage_temp]
                    sleep_label[subject] = stage_temp
                else:
                    stage_temp = [x + '1' for x in stage_temp]
                    sleep_label[subject] = stage_temp
        return data_dict ,sleep_label
    

