import warnings
import datetime
import os
import pickle
from pathlib import Path
from random import sample
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score, cross_validate, train_test_split, cross_val_predict
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from aa_features_pre import aa_features_final
from physicochemical_property_cal import engineered_properties_cal
from fasta_transfer import sequences_to_fasta
warnings.filterwarnings('ignore')

def Partial_d(target):
    partial_result = []; partial_names = []
    for target_f in ['fraction_C', 'gravy']:
        original_data = total_aa.copy()
        target_d = original_data[target_f].values
        for perc in [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]:
            value_perc = np.percentile(target_d, perc)
            original_data = total_aa.copy()
            original_data[target_f] = value_perc

            pred_new = []
            for i in range(10):
                filename = "my_model_"+target+'_'+str(i+1)+".pickle"
                loaded_model = pickle.load(open(filename, "rb"))
                #print(original_data[target_f])

                # for model prediction
                data_v = original_data.values[:, 2:]
                pred_new.append(loaded_model.predict_proba(data_v)[:,1])

            prob_new = (np.array(pred_new[0]) + np.array(pred_new[1]) + np.array(pred_new[2]) + np.array(pred_new[3]) + np.array(pred_new[4]) + np.array(pred_new[5]) + np.array(pred_new[6]) + np.array(pred_new[7]) + np.array(pred_new[8]) + np.array(pred_new[9]))/10
            #print(prob_new)
            partial_result.append(prob_new)
            partial_names.append(value_perc)
    partial_df = pd.DataFrame(partial_result).T
    partial_df.columns = partial_names
    return partial_df


def predict_proteome(name):
    data = pd.read_csv(name)
    data['Sequence'] = data['Sequence'].str.replace('X', '')
    data['Sequence'] = data['Sequence'].str.replace('U', '')
    data['Sequence'] = data['Sequence'].str.replace('B', '')
    # data processing
    # data : a data frame with column 'Sequence'
    sequences_to_fasta(data, 'Entry', 'new.fasta')
    features_aa = aa_features_final('new.fasta')
    #print('new_aa columns: ', new_aa.columns)
    global total_aa
    total_aa = features_aa.drop(['protein_name', 'sequence', 'HydroPhobicIndex'], errors='ignore')
    new_aa = total_aa.values[:, 2:]
    #print(new_aa.columns)

    for target in ['SG', 'PB', 'PBSG']:
        pred_new = []
        for i in range(10):
            # load model
            filename = "my_model_"+target+'_'+str(i+1)+".pickle"
            loaded_model = pickle.load(open(filename, "rb"))

            # predict new data
            pred_new.append(loaded_model.predict_proba(new_aa)[:,1])
            #print(loaded_model.predict_proba(new_w2v)[:,1])

        # Calculate the metrics for this fold
        prob_new = (np.array(pred_new[0]) + np.array(pred_new[1]) + np.array(pred_new[2]) + np.array(pred_new[3]) + np.array(pred_new[4]) + np.array(pred_new[5]) + np.array(pred_new[6]) + np.array(pred_new[7]) + np.array(pred_new[8]) + np.array(pred_new[9]))/10

        features_aa['ave_proba_'+target] = prob_new

        '''
        # ICE results for two key features
        partial_df = Partial_d(target)
        partial_df.to_csv('partial_dependence_proteome'+target+'.csv', index=False)'''

    features_aa.to_csv(name+'_pred_check_20241125.csv', index=False)
    print(features_aa.head())

    return

predict_proteome('uniprot_human_proteome.csv')