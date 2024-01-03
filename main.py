from platform import node
from random import sample
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras import backend as K
from sklearn.model_selection import StratifiedKFold
from tensorflow.core.protobuf.config_pb2 import OptimizerOptions
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import gc
from utils import *
from ST_HAG import ST_HAG_ADHD
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix



os.environ['PYTHONHASHSEED']=str(42)
random.seed(42)
np.random.seed(42)
# tf.random.set_seed(42)
tf.set_random_seed(42)



def FC_adj(data):
    """Extracting the functional connectivity matrix as our adjacent matrix

    Args:
        data: input fMRI time-seires

    Returns:
        feat: functional connectivity matrix
    """
    feature = extract_feature(data)
    feature = np.array(feature)
    return feature

def extract_feature(data):
    feat = []
    for i in range(data.shape[0]):
        fc= Persion_matrix(data[i])
        fc = pd.DataFrame(fc)
        fc = fc.fillna(0)
        feat.append(fc.values)
    return feat

def Persion_matrix(data):
    FC = np.corrcoef(data,rowvar=0)
    return FC



def data_augment(data, label, pheno, crops):
    """    performing data cropping for data augmentation
    Args:
        data: fMRI time-series data
        label: corresponding labels of input fMRI data
        pheno: corresponding phenotypic information of input fMRI data （age，sex，IQ）
        crops: cropping numbers of one data sequence

    Returns:
        augmented fMRI data, labels and phenotypic informations
    """
    augment_label = []
    augment_data = []
    augmented_pheno = []
    sk_data = []
    sk_label = []
    for i in range(len(data)):
        max = data[i].shape[0]
        if max >= 100:
            sk_data.append(data[i])
            sk_label.append(label[i])

            range_list = range(90 + 1, int(max))
            random_index = random.sample(range_list, crops)
            for j in range(crops):
                r = random_index[j]
                # r = random.randint(90,int(max))
                augment_data.append(data[i][r - 90:r])
                augment_label.append(label[i])
                augmented_pheno.append(pheno[i])

    return np.array(augment_data), np.array(augment_label), np.array(augmented_pheno), np.array(sk_data), np.array(
        sk_label)
    

def main():
    crops = 10
    ###### load dataset
    feats = np.load('Data/ADHD/ADHD_feats_norm.npy', allow_pickle=True)
    labels = np.load('Data/ADHD/ADHD_labels_norm.npy')
    phenos = np.load('Data/ADHD/ADHD_phenos_norm.npy', allow_pickle=True)
    
    augment_feats, augment_labels, augment_pheno, sk_data, sk_label = data_augment(feats,labels,
                                                                                                  phenos, crops)

    ####### Cross validation
    CV_count = 30
    sub_acc_mean = []
    sub_sensitive_mean = []
    sub_specificity_mean = []
    sub_auc_mean = []

    for i_count in range(CV_count):

        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2 ** (i_count + 1))
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2 ** (1 + 1))

        j_count = 0

        sub_acc = []
        sub_sensitive = []
        sub_specificity = []
        sub_auc = []

        for train_index, test_index in kf.split(sk_data, sk_label):
            

            aug_x_train_list = []
            aug_x_test_list = []
            true_test_label_list = []
            count = 0


            for i in range(sk_data.shape[0]):
                if i in train_index:
                    for k in range(crops):
                        aug_x_train_list.append(count)
                        count = count + 1
                    
                else:
                    true_test_label_list.append(i)
                    for k in range(crops):
                        aug_x_test_list.append(count)
                        count = count + 1

            # training data
            x_train = augment_feats[aug_x_train_list]
            y_train = augment_labels[aug_x_train_list]
            x_train_pheno = augment_pheno[aug_x_train_list]
            
            node_num, feature_dim = x_train.shape[2],x_train.shape[1]


            # testing data
            x_test = augment_feats[aug_x_test_list]
            y_test = augment_labels[aug_x_test_list]
            x_test_pheno = augment_pheno[aug_x_test_list]
            
            true_test_label = sk_label[true_test_label_list]

            Graph_connect = FC_adj(sk_data[train_index]) 
            Graph_connect = Graph_connect.mean(axis=0)
            for i in range(Graph_connect.shape[0]):
                for j in range(Graph_connect.shape[1]):
                    if Graph_connect[i][j] > 0.1:
                        Graph_connect[i][j] = 1
                    else:
                        Graph_connect[i][j] = 0
            print(Graph_connect.shape)
            print(Graph_connect[np.nonzero(Graph_connect)].shape)
                        
            zero_padding = np.zeros(shape=(node_num,node_num))
            temporal_connect = np.eye(node_num)

            A1 = np.concatenate((Graph_connect,temporal_connect,zero_padding),axis=1)
            A2 = np.concatenate((temporal_connect,Graph_connect,temporal_connect),axis=1)
            A3 = np.concatenate((zero_padding,temporal_connect,Graph_connect),axis=1)
            A = np.concatenate((A1,A2,A3),axis=0).astype(np.int16)
            
            A_train = np.repeat(A[None,:], x_train.shape[0], axis=0)
            A_test = np.repeat(A[None,:], x_test.shape[0], axis=0)     
            
            print(A_train.shape)
            
            # shuffle
            index = [i for i in range(x_train.shape[0])]
            random.shuffle(index)
            x_train_pheno = x_train_pheno[index]
            y_train = y_train[index]
            x_test_pheno = x_test_pheno.reshape(-1,4)
            

            x_train = x_train[index].swapaxes(2, 1)
                
            model = ST_HAG_ADHD(feature_dim,node_num)
            # model.summary()
         
            early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=2)

            model.fit(x=[x_train, A_train,x_train_pheno], y=y_train, epochs=30, batch_size=128,
                        validation_split=0.05,callbacks=[early_stopping])

            x_test = x_test.swapaxes(2, 1)
            predict = model.predict(x=[x_test, A_test,x_test_pheno])

            pred2 = []

            pred_score = []
            for i in range(len(test_index)):

                ADHD_subject_score = predict[i * crops:i * crops + crops]
                pred_score.append(np.sum(ADHD_subject_score) / crops)
                if np.sum(ADHD_subject_score) / crops >= 0.5:
                    pred2.append(1)
                else:
                    pred2.append(0)


            [[TN, FP], [FN, TP]] = confusion_matrix(true_test_label, pred2).astype(float)
            specificity = TN / (FP + TN)
            sensivity = recall = TP / (TP + FN)
            auc = roc_auc_score(true_test_label, pred_score)


            sub_acc.append(accuracy_score(true_test_label, pred2))
            sub_auc.append(auc)
            sub_specificity.append(specificity)
            sub_sensitive.append(sensivity)

            print("sub accuracy: " + str(accuracy_score(true_test_label, pred2)))
            print("sub auc: " + str(sub_auc))
            print("sub specificity: " + str(sub_specificity))
            print("sub sensivity: " + str(sub_sensitive))

            j_count = j_count + 1
            K.clear_session()
            # tf.reset_default_graph()
            del [A,A1,A2,A3,A_train,A_test,Graph_connect,temporal_connect,zero_padding]
            gc.collect()

        sub_acc_mean.append(np.mean(sub_acc))
        sub_auc_mean.append(np.mean(auc))
        sub_specificity_mean.append(np.mean(sub_specificity))
        sub_sensitive_mean.append(np.mean(sub_sensitive))

        print("sub acc mean: " + str(sub_acc_mean))
        print("sub auc_mean: " + str(sub_auc_mean))
        print("sub specificity_mean: " + str(sub_specificity_mean))
        print("sub sensitive_meann: " + str(sub_sensitive_mean))
        
if __name__ == '__main__':
    main()