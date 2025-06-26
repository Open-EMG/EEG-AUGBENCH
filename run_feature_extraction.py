import numpy as np
import os
from pywt import wavedec
from define_compute import *

def extract_features_all(X,sfreq):
    epochs, channel, sample_length = X.shape
    features = np.zeros((epochs, 3*channel*4))
    for i in range(epochs):
        delta, theta, alpha, beta = wavedec(X[i,:,:], 'db1', level=3)
        temp_compute_energy_dwt = compute_energy_dwt(delta, theta, alpha, beta)
        temp_compute_variance_dwt = compute_variance_dwt(delta, theta, alpha, beta)
        temp_compute_TK_energy_dwt = compute_TK_energy_dwt(delta, theta, alpha, beta)
        features[i, :] = np.concatenate((temp_compute_variance_dwt, temp_compute_energy_dwt, temp_compute_TK_energy_dwt),axis=0)
    return features

def feature_extraction(aug_name, rate, fs, raw_data_root, augmented_root, feature_root):
        
    for i in range(1, 25):
        subject_id = f"{i:02d}"
        print(f"[{aug_name}] Processing subject {subject_id}")
        
        if aug_name == "no":
            subject_dir = os.path.join(raw_data_root, f"chb{subject_id}")
        else:
            subject_dir = os.path.join(augmented_root, aug_name, str(rate), f"chb{subject_id}")
        file_list = os.listdir(subject_dir)

        for file_name in file_list:
            if aug_name == "no":
                feature_dir = os.path.join(feature_root, aug_name, f"chb{subject_id}")
            else:
                feature_dir = os.path.join(feature_root, aug_name, str(rate), f"chb{subject_id}")
            os.makedirs(feature_dir, exist_ok=True)

            feature_filename = f"feature_{file_name}"
            feature_path = os.path.join(feature_dir, feature_filename)

            if os.path.exists(feature_path):
                print(f"  └── Skipping existing: {feature_filename}")
                continue

            # 读取数据
            file_path = os.path.join(subject_dir, file_name)
            # import pdb; pdb.set_trace()
            data = np.load(file_path)
            eeg = data["EEG"]
            labels = data["label"]
            
            if aug_name is not "no":
                eeg = np.transpose(eeg, (0, 2, 1))  # 维度调整

            # 特征提取
            features = extract_features_all(eeg, 256)

            # 保存特征
            np.savez(feature_path, feature=features, label=labels)

