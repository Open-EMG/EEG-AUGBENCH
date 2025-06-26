import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, roc_curve, auc
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity


def classification(aug_method, rate, cls_method, feature_root, result_root):

    all_result = np.zeros((24, 5))
    save_road_all = os.path.join(result_root, aug_method, str(rate))
    if not os.path.exists(save_road_all):
        os.makedirs(save_road_all)
    save_road_file = save_road_all + '/result_' + cls_method.upper() + '.npz'
    if os.path.exists(save_road_file):
        print("File already exists, skipping...")
        return
    for i in range(1, 25):
        print(i)
        subject_id = f"{i:02d}"
        print(f"[{aug_method}] Processing subject {subject_id}")
        feature_no_dir =  os.path.join(feature_root, 'no', f"chb{subject_id}")
        files = os.listdir(feature_no_dir)
        all_index = np.array(range(0, len(files)))
        mean_acc_subject = np.zeros((len(files)))
        mean_pre_subject = np.zeros_like(mean_acc_subject)
        mean_recall_subject = np.zeros_like(mean_acc_subject)
        mean_f1_subject = np.zeros_like(mean_acc_subject)
        mean_auc_subject = np.zeros_like(mean_acc_subject)

        
        for j in range(len(files)):

            
            all_train_y = None
            all_train_feature = None
            test_index = j
            test_partial_road = feature_no_dir + '/' + files[j]  # classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
            test_tmp = np.load(test_partial_road)
            X_test = test_tmp['feature']
            Y_test = test_tmp['label']
            train_index = set(all_index) - set(np.array([j]))
 
            for m in range(len(train_index)):
                train_partial_road_no = feature_no_dir + '/' + files[list(train_index)[m]]
                train_tmp = np.load(train_partial_road_no)
                X_training = train_tmp['feature']
                Y_training = train_tmp['label']
                num_seizure = np.shape(np.where(Y_training==1)[0])[0]

                if aug_method !='no':

                    aug_road_file = os.path.join(feature_root, aug_method, str(rate),  f"chb{subject_id}", files[list(train_index)[m]])
                  
                    aug_temp = np.load(aug_road_file)
                    X_training_aug = aug_temp['feature']
                    Y_training_aug = aug_temp['label']
                    X_training = np.concatenate((X_training, X_training_aug[0:num_seizure*rate, :]), axis=0)
                    Y_training = np.concatenate((Y_training, Y_training_aug[0:num_seizure*rate]), axis=0)
                else:
                    seizure_index = np.where(Y_training == 1)[0]
                    current_num_sei = np.shape(seizure_index)[0]
                    current_num_no = np.shape(np.where(Y_training == 0)[0])[0]
                  
                    for rate_i in range(rate):
                        X_training = np.concatenate((X_training, X_training[seizure_index, :]))
                        Y_training = np.concatenate((Y_training, Y_training[seizure_index]))
                        current_num_sei = np.shape(np.where(Y_training == 1)[0])[0]
                if m == 0:
                    X_training_all = X_training
                    Y_training_all = Y_training
                else:
                    X_training_all = np.concatenate((X_training_all, X_training), axis=0)
                    Y_training_all = np.concatenate((Y_training_all, Y_training), axis=0)
            scaler = StandardScaler()
            X_training_all = scaler.fit_transform(X_training_all)
            X_test = scaler.transform(X_test)
      
            if cls_method.upper() == 'KNN':
                clf = KNeighborsClassifier()
            elif cls_method.upper() == 'LRC':
                clf = LogisticRegression(penalty='l2')
            elif cls_method.upper() == 'RFC':
                clf = RandomForestClassifier(n_estimators=200)
            elif cls_method.upper() == 'DTC':
                clf = tree.DecisionTreeClassifier()
            elif cls_method.upper() == 'ADABOOST':
                clf = AdaBoostClassifier()


            # Train classifier
            clf.fit(X_training_all, Y_training_all)


            predictions = clf.predict(X_test)
            mean_acc_subject[j] = accuracy_score(Y_test, predictions)
            mean_pre_subject[j] = specificity_score(Y_test, predictions)
            mean_recall_subject[j] = recall_score(Y_test, predictions)
            mean_f1_subject[j] = f1_score(Y_test, predictions)
            fpr, tpr, thresholds = roc_curve(Y_test, predictions)
            mean_auc_subject[j] = auc(fpr, tpr)
        all_result[i - 1, 0] = np.mean(mean_acc_subject)
        all_result[i - 1, 1] = np.mean(mean_pre_subject)
        all_result[i - 1, 2] = np.mean(mean_recall_subject)
        all_result[i - 1, 3] = np.mean(mean_f1_subject)
        all_result[i - 1, 4] = np.mean(mean_auc_subject)

    np.savez(save_road_file, all_result=all_result)


