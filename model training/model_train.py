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


from data_processing import aa_features
from physicochemical_property_cal import engineered_properties_cal
from fasta_transfer import sequences_to_fasta
warnings.filterwarnings('ignore')


###################################################################################
###################################################################################
###################################################################################
###################################################################################

def fold_validation(rf, dataset_w2v, pdb_data):
    plt.figure(figsize=(2.8, 1.3))
    plt.rcParams['font.family'] = 'Times New Roman'
    X_val = pdb_data[:, :-1]; y_val = pdb_data[:, -1]
    # Define the number of folds
    n_splits = 10
    # Create a KFold object
    kf = KFold(n_splits=n_splits)
    # Initialize lists to store the scores for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    auc_scores = []
    accuracy_scores_val = []
    pred_new = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    precisions = []
    mean_recall = np.linspace(0, 1, 100)
    pr_aucs = [] # list to store the AUC values for the precision-recall curve

    # Initialize a list to store the feature importances for each fold
    feature_importances = []

    # Split the data into training and test sets for each fold
    num = 0 
    for train_index, test_index in kf.split(dataset_w2v):
        num += 1
        # Get the training and test sets for this fold
        X_train, X_test = dataset_w2v[train_index, :-1], dataset_w2v[test_index, :-1]
        y_train, y_test = dataset_w2v[train_index, -1], dataset_w2v[test_index, -1]

        # Train the model on the training set
        rf.fit(X_train, y_train)

        filename = "my_model_"+target+'_'+str(num)+".pickle"
        # save model
        pickle.dump(rf, open(filename, "wb"))

        # Append the feature importances for this fold to the list
        feature_importances.append(rf.feature_importances_)
        #feature_imp(rf)

        # Make predictions on the test set
        y_pred = rf.predict(X_test)
        # Calculate the metrics for this fold
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))
        
        probas_ = rf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        '''plt.plot(fpr, tpr,
                 lw=2,
                 color='lightgray')'''

        precision_fold, recall_fold, _ = precision_recall_curve(y_test, probas_[:, 1])
        
        sort_index = np.argsort(recall_fold)
        recall_fold = recall_fold[sort_index]
        precision_fold = precision_fold[sort_index]
    
        precisions.append(np.interp(mean_recall, recall_fold[::-1], precision_fold[::-1]))
        pr_auc = auc(recall_fold[::-1], precision_fold[::-1]) # calculate AUC for precision-recall curve
        pr_aucs.append(pr_auc) # add AUC value to list
        # predict pdb data
        y_pred_val = rf.predict(X_val)
        accuracy_scores_val.append(accuracy_score(y_val, y_pred_val))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    auc_df = pd.DataFrame([mean_fpr, mean_tpr]).T
    auc_df.columns = ['mean_fpr', 'mean_tpr']
    auc_df.to_csv(target+'_auc_df.csv')

    '''plt.plot(mean_fpr, mean_tpr,
             label=r'Mean ROC (AUC=%0.2f)' % (np.mean(aucs)),
             lw=2,
             color='black')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()
    plt.close()'''

    '''feature_importances_df = pd.DataFrame(feature_importances, columns=features_X)
    feature_importances_df.to_csv(target+'feature_importances.csv')'''
    # Calculate the mean and standard deviation of the scores for each metric
    # Print the results
    print(f'Accuracy: {np.mean(accuracy_scores):.2f} (+/- {np.std(accuracy_scores):.2f})')
    print(f'Precision: {np.mean(precision_scores):.2f} (+/- {np.std(precision_scores):.2f})')
    print(f'Recall: {np.mean(recall_scores):.2f} (+/- {np.std(recall_scores):.2f})')
    print(f'F1-score: {np.mean(f1_scores):.2f} (+/- {np.std(f1_scores):.2f})')

    print(f'AUC: {np.mean(aucs):.2f} (+/- {np.std(aucs):.2f})')
    print(f'PR AUC: {np.mean(pr_aucs):.2f} (+/- {np.std(pr_aucs):.2f})') # print mean and standard deviation of AUC values for precision-recall curve

    print(f'Accuracy val: {np.mean(accuracy_scores_val):.2f} (+/- {np.std(accuracy_scores_val):.2f})')
    return


def overall_prep_aa(RNA_granule_type, tier, neg_sample_n):
    # prepare data set
    aa_data = pd.read_csv('total_aa.csv')
    print(aa_data.columns)
    sg_aa = aa_data[(aa_data['human_Tier'] <= tier) & (aa_data['data_class'] == RNA_granule_type)]
    proteome_neg_aa = aa_data[(aa_data['data_class'] == 'proteome_neg')]
    pdb_aa = aa_data[(aa_data['data_class'] == 'pdb')]
    #print(sg_aa, proteome_neg_aa)
    sg_aa = sg_aa.drop(['protein_name', 'human_Tier', 'data_class', 'sequence'], axis=1)
    proteome_neg_aa = proteome_neg_aa.drop(['protein_name', 'human_Tier', 'data_class', 'sequence'], axis=1)

    global features_X
    features_X = list(proteome_neg_aa.columns[:-1])
    pdb_aa = pdb_aa.drop(['protein_name', 'human_Tier', 'data_class', 'sequence'], axis=1).values
    print(sg_aa.shape, proteome_neg_aa.shape, pdb_aa.shape)

    # set training set, testing set, val_set
    sample_neg_aa = proteome_neg_aa.sample(n=int(neg_sample_n*sg_aa.shape[0]), random_state=1)
    dataset_aa = pd.concat([sg_aa, sample_neg_aa], axis=0, ignore_index=True)
    print(dataset_aa.columns)
    dataset_aa = shuffle(dataset_aa, random_state=2).values

    return dataset_aa, pdb_aa


def feature_imp(model):
    # get feature importances
    importances = model.feature_importances_

    # sort the features by importance
    indices = np.argsort(importances)[::-1]

    # print the feature ranking
    print("Feature ranking:")
    for f in range(len(features_X)):
        print(f"{f + 1}. feature {features_X[indices[f]]} ({importances[indices[f]]})")
    return


def model_train():
    aa_features()
    global target, dataset_aa
    target = 'SG' # 'PBSG'
    tier = 1 # tier 1 for 'PBSG'/'SG'; tier 2 for 'PB'
    sample_time_proteome_neg = 1
    dataset_aa, pdb_aa = overall_prep_aa(target, tier, sample_time_proteome_neg)
    print(dataset_aa.shape, pdb_aa.shape)

    # RF classifier
    # Create a RF classifier
    rf = RandomForestClassifier(n_estimators=2000, random_state=1, class_weight='balanced', criterion='entropy', n_jobs=20)
    fold_validation(rf, dataset_aa, pdb_aa)

####
model_train()
