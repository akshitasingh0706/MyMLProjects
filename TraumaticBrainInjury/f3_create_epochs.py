import numpy as np
from f2_combine_data import *

data_dict, sleep_dict = combine_data()

class create_epoches:
    def __init__(self,
                epoch_len = 30,
                species = 'human',
                fsh = 200):
        self.epoch_len = epoch_len
        self.species = species
        self.fsh = fsh


    def run(self, data , species):  
        fs = 256 # for mice  
        epoch_data = {}
        for subject in data:
            if species == 'human':
                fs = self.fsh
                
            num_epoch = int(np.shape(data[subject])[1]/(self.epoch_len * fs))
            temp_data = []
            for idx in range(num_epoch):
                start = idx * fs * self.epoch_len
                stop =  start +fs * self.epoch_len
                temp_data.append(data[subject][:,start : stop])
            epoch_data[subject] = np.transpose(temp_data, (1, 2, 0))
        return epoch_data

epochs = create_epoches()
control_epoch_human = create_epoches.run(data_dict['control_human'], 'human')
tbi_epoch_human = create_epoches.run(data_dict['tbi_human'], 'human')
# control_epoch_mice = create_epoches.run(data_dict['control_mice'], 'mice')
# tbi_epoch_mice = create_epoches.run(data_dict['tbi_mice'], 'mice')