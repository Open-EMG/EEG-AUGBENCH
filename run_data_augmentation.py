
import numpy as np
import os
import pandas as pd
from pywt import wavedec
from augmentation import *
from sklearn.preprocessing import StandardScaler

AUG_METHODS = {
    "jitter":   jitter,
    "scaling":  scaling,
    "permutation": permutation,
    "magwarp":      magnitude_warp,
    "timewarp":     time_warp,
    "windowslice":  window_slice,
    "windowwarp":   window_warp,
    "rgw"    :  random_guided_warp,
    "rgws"   :  random_guided_warp_shape,
    "scaling_multi":      scaling_multi,
    "windowwarp_multi":   window_warp_multi,
}


def get_aug_by_name(name):
    
    if name not in AUG_METHODS.keys():
        raise ValueError(
            "The name specified '%s' is not a valid augmentation method.\n\
            Valid methods are: [%s]" % (name, str(AUG_METHODS.keys())))
    return AUG_METHODS[name]

def data_augmentation(aug_method, rate, raw_data_root, augmented_root):

    for i in range(1, 25):
        subject_id = f"{i:02d}"  
        subject_dir = os.path.join(raw_data_root, f"chb{subject_id}")
        file_list = os.listdir(subject_dir)
        file_indices = np.arange(len(file_list))

        for file_idx, file_name in enumerate(file_list):
            aug_save_dir = os.path.join(augmented_root, aug_method, str(rate), f"chb{subject_id}")
            os.makedirs(aug_save_dir, exist_ok=True)

 
            save_path = os.path.join(aug_save_dir, file_name)
            if os.path.exists(save_path) is True:
                print(f"[INFO] Skipping existing file: {file_name}")
                continue
            
            print(f"[INFO] Processing file: {file_name}")
            file_path = os.path.join(subject_dir, file_name)
            data_npz = np.load(file_path)

            eeg_data = data_npz['EEG']                     
            labels = data_npz['label']                      
            eeg_data = np.transpose(eeg_data, (0, 2, 1))    

            seizure_indices = np.where(labels == 1)[0]
            non_seizure_indices = np.where(labels == 0)[0]

            aug_function = get_aug_by_name(aug_method)


            raw_X = eeg_data[seizure_indices, :, :]    
            raw_y = labels[seizure_indices]         

            aug_X = np.zeros((rate * len(seizure_indices), raw_X.shape[1], raw_X.shape[2]))
            aug_y = np.zeros((rate * len(seizure_indices)))

            for r in range(rate):
                start = r * len(seizure_indices)
                end = (r + 1) * len(seizure_indices)
                aug_X[start:end] = aug_function(raw_X, raw_y)
                aug_y[start:end] = 1
            np.savez(save_path, EEG=aug_X, label=aug_y)