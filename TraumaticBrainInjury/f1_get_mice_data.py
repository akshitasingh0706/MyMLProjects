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

class get_mice_data:
    def __init__(self,
        data_folder: str = root_path + '/Mice/data',
        data_label: list = ['data1','data2'],
        control_data1: list = ['Sham102_BL5','Sham103_BL5'],
        tbi_data1: list = ['TBI101_BL5','TBI102_BL5'],
        control_data2: list = ['m010_TPE01_BaselineDay2_sham', 'm010_TPE03_BaselineDay2_sham'],
        tbi_data2: list = ['XVZ2FYAQH8WVIUC','XVZ2FYATE84MSWI'],    
        fs = 256,  
        
        freq_band: str = 'normal',   # 'delta','theta','alpha','sigma','beta','gama','normal'
        filter_domain: str = ['time', 'freq'],
        
        sleep_stage: str = 'NR', 
        epoch_len: int = 28,
        duration: int = 19,
        features: list = ['absolute_power','relative_power','slow_fast','frequency amplitude asymmetry',
                        'phase synchrony','coherence','hjorth','spectral_entropy','phase amplitude coupling']
        ):
        self.data_folder = data_folder
        self.data_label = data_label
        self.control_data1 = control_data1
        self.tbi_data1 = tbi_data1
        self.control_data2 = control_data2
        self.tbi_data2 = tbi_data2
        self.fs = fs

        self.freq_band = freq_band
        self.filter_domain = filter_domain
        self.time_params = [0.5, 50] if filter_domain == 'time' else None

        self.sleep_stage = sleep_stage
        self.epoch_len = epoch_len
        self.duration = duration
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
        control, tbi = {}, {}
        subjects = self.organize_data(data_label, class_label)
        if data_label == 'data2':
            temp_df = pd.read_csv(self.data_folder +'/' + 'Channel_Contents'+'.txt',header=None, index_col=None,error_bad_lines=False)
            groups =  temp_df.iloc[1:,0].str.split(expand=True)
            groups.columns = ['File','EMG','EEG','EEG','Group','Label']
        
        # 1) Load staging files (required for future preproccing steps in raw data)
        for subject in subjects:
            if data_label == 'data2':
                df = pd.read_csv(self.data_folder + '/' + subject +'_Stages.txt',  error_bad_lines=False,sep='delimiter', header=None)
                if class_label == 'control':
                    nu=16
                    df = df.drop(df.index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]])
                elif class_label == 'ybi':
                    nu=0
                loc  = df.loc[nu]
                stage_data = x.str.split('\t',expand=True)
                for j in range(nu,len(df)+nu-1):
                    loc = df.loc[j]
                    temp = loc.str.split('\t',expand=True)
                    stage_data = stage_data.append(temp)
                stage_data.columns = stage_data.iloc[0]
                stage_data = stage_data[stage_data.iloc[:,3] != 'Stage']
                stage_data = stage_data.iloc[2:]
            elif data_label == 'data1':
                df = pd.read_csv(self.data_folder +'/'+ subject +'_Stages.csv', sep='delimiter', header=None)
                df = df.drop(df.index[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]])
                loc  = df.loc[18]
                stage_data  = loc.str.split(',',expand=True)
                for j in range(19,len(df)+18):
                    loc = df.loc[j]
                    temp = loc.str.split(',',expand=True)
                    stage_data = stage_data.append(temp)
                stage_data.columns = stage_data.iloc[0]
                stage_data = stage_data[stage_data.iloc[:,3] != 'Stage']
                stage_data = stage_data.iloc[2:]
            
            # 2) Load raw EEG data from the .fif and/or .edf files
            raw_data = mne.io.read_raw_edf(self.data_folder + '/' + subject + ".edf", preload=True,verbose=None)
            
            # Filter the data based on the time domain (if specified)
            if self.filter_domain == 'time':
                filtered_data = raw_data.copy()
                filtered_data.load_data().filter(self.time_params)    
                raw_data = filtered_data

            # 3) Process the raw data
            data = []
            j = 0
            for i in range(len(stage_data)):
                if stage_data.iloc[i,2] == self.sleep_stage:
                    start = i * self.fs * 4
                    stop = i * self.fs * 4 + (self.fs * 4)
                    data[j: j + 4 * self.fs] = (raw_data[start : stop])
                    j = j + 4 * self.fs
            
            data = data - np.mean(data)
            if class_label == 'control':
                name = 'control_' + self.sleep_stage + subject
                control[name] = data.reshape((1, len(data)))
            else:
                name = 'tbi_'+ self.sleep_stage + subject
                tbi[name] = data.reshape((1, len(data)))

        new_list = [val for n, val in enumerate(subjects)]
        return control, new_list if class_label == 'control' else tbi, new_list
