import numpy as np
from f1_get_human_data import *
from f1_get_mice_data import *

data_labels = ['data1', 'data2']
class_labels = ['control', 'human']


def combine_data():
    final_dict = {}
    human_data = get_human_data()
    # mice_data = get_mice_data()

    control_human_data_dict_all, control_human_sleep_dict_all = {}, {}
    tbi_human_data_dict_all, tbi_human_sleep_dict_all = {}, {}
    # control_mice_data_dict_all, control_mice_sleep_dict_all = {}, {}
    # tbi_mice_data_dict_all, tbi_mice_sleep_dict_all = {}, {}
    for dlabel in data_labels:
        control_human_data_dict, control_human_sleep_dict = human_data.run(dlabel, 'control')
        tbi_human_data_dict, tbi_human_sleep_dict = human_data.run(dlabel, 'tbi')

        for subject in control_human_data_dict.keys():    
            control_human_data_dict_all[subject] = control_human_data_dict[subject]
            control_human_sleep_dict_all[subject] = control_human_sleep_dict[subject]
        for subject in tbi_human_data_dict.keys():    
            tbi_human_data_dict_all[subject] = tbi_human_data_dict[subject]
            tbi_human_sleep_dict_all[subject] = tbi_human_sleep_dict[subject]
        
        # uncomment once debugged
        # control_mice_data_dict, control_mice_sleep_dict = mice_data.run(dlabel, 'control')
        # tbi_mice_data_dict, tbi_mice_sleep_dict = mice_data.run(dlabel, 'tbi')

        # for subject in control_mice_data_dict.keys():    
        #     control_mice_data_dict_all[subject] = control_mice_data_dict[subject]
        #     control_mice_sleep_dict_all[subject] = control_mice_sleep_dict[subject]
        # for subject in tbi_human_data_dict.keys():    
        #     tbi_mice_data_dict_all[subject] = tbi_mice_data_dict[subject]
        #     tbi_mice_sleep_dict_all[subject] = tbi_mice_sleep_dict[subject]
    
    final_data_dict = {
        'control_human': control_human_data_dict_all,
        'tbi_human': tbi_human_data_dict_all,
        # 'control_mice': control_mice_data_dict_all,
        # 'tbi_mice': tbi_mice_data_dict_all
    }

    final_sleep_dict = {
        'control_human': control_human_sleep_dict_all,
        'tbi_human': tbi_human_sleep_dict_all,
        # 'control_mice': control_mice_sleep_dict_all,
        # 'tbi_mice': tbi_mice_sleep_dict_all
    }

    return final_data_dict, final_sleep_dict
        