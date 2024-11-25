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
from sklearn.metrics import roc_curve, auc


from aa_features_pre import aa_features_final
from physicochemical_property_cal import engineered_properties_cal
from fasta_transfer import sequences_to_fasta
warnings.filterwarnings('ignore')


###################################################################################
###################################################################################
###################################################################################
###################################################################################
def add_columns_combine(df1_data, df2_data, df3_data, df4_data, df5_data):
    def add_columns(df0_d, data_class):
        #print(df0.shape, df0_d.shape)
        df0_d = df0_d[['Entry', 'human_Tier', 'Sequence']].copy()
        df0_d.columns = ['protein_name', 'human_Tier', 'Sequence']
        #df0['protein_name'] = np.array(df0_d['Entry'].values.tolist())
        #df0['human_Tier'] = np.array(df0_d['human_Tier'].values.tolist())
        df0_d['data_class'] = data_class
        df0_d['label'] = 1
        return df0_d
    #print(df3, df3_data)
    #print(df1.shape, df1_data.shape)
    df_SG = add_columns(df1_data, 'SG')
    df_PB = add_columns(df2_data, 'PB')
    df_PBSG = add_columns(df3_data, 'PBSG')

    df4 = df4_data[['Entry', 'Sequence']].copy()
    df4.columns = ['protein_name', 'Sequence']
    df4['human_Tier'] = 0
    df4['data_class'] = 'proteome_neg'
    df4['label'] = 0

    df5 = df5_data[['Entry', 'Sequence']].copy()
    df5.columns = ['protein_name', 'Sequence']
    df5['human_Tier'] = 0
    df5['data_class'] = 'pdb'
    df5['label'] = 0

    df_total = pd.concat([df_SG, df_PB, df_PBSG, df4, df5], axis=0)
    return df_total

def find_difference_name(df1, df2):
    set1 = set(df1["Entry"])
    set2 = set(df2["Entry"])
    diff_set = set2 - set1
    diff_df = df2[df2["Entry"].isin(diff_set)]
    return diff_df


def aa_features():
    # read RNA granule file
    SG_data_positive = pd.read_excel("RNA_granule_data.xlsx", sheet_name='SG')
    PB_data_positive = pd.read_excel("RNA_granule_data.xlsx", sheet_name='PB')
    PBSG_data_positive = pd.read_excel("RNA_granule_data.xlsx", sheet_name='PBSG').dropna().reset_index(drop=True)
    print('shape of SG, PB, PBSG with all tiers: ', SG_data_positive.shape, PB_data_positive.shape, PBSG_data_positive.shape)


    # read human proteome
    proteome = pd.read_csv('uniprot_human_proteome.csv')
    proteome['Sequence'] = proteome['Sequence'].str.replace('X', '')
    proteome['Sequence'] = proteome['Sequence'].str.replace('U', '')
    proteome['Sequence'] = proteome['Sequence'].str.replace('B', '')
    print('shape of original proteome: ', proteome.shape)


    # read pdb files
    pdb_data = pd.read_csv('pdb30.csv')
    pdb_data.columns = ['PDB', 'Entry', 'seq_len', 'Sequence']

    pdb_data1 = find_difference_name(SG_data_positive, pdb_data)
    pdb_data2 = find_difference_name(PB_data_positive, pdb_data1)
    pdb_data3 = find_difference_name(PBSG_data_positive, pdb_data2)
    pdb_data_neg = find_difference_name(proteome, pdb_data3)
    print('shape of pdb proteins: ', pdb_data_neg.shape)


    # proteome_neg
    neg_1 = find_difference_name(SG_data_positive, proteome)
    neg_2 = find_difference_name(PB_data_positive, neg_1)
    proteome_neg = find_difference_name(PBSG_data_positive, neg_2)
    print('shape of negative proteome: ', proteome_neg.shape)

    # concat
    total_data = add_columns_combine(SG_data_positive, PB_data_positive, PBSG_data_positive, proteome_neg, pdb_data_neg)
    total_data['Sequence'] = total_data['Sequence'].str.replace('X', '')
    total_data['Sequence'] = total_data['Sequence'].str.replace('U', '')
    total_data['Sequence'] = total_data['Sequence'].str.replace('B', '')
    total_data.to_csv('total_data.csv', index=False)
    print('total data column:', total_data.columns)
    ###################################################################################
    # aa features
    sequences_to_fasta(total_data, 'protein_name', 'total_data.fasta')
    total_aa = aa_features_final('total_data.fasta')
    total_aa[['human_Tier', 'data_class', 'label']] = total_data[['human_Tier', 'data_class', 'label']].values
    total_aa.to_csv('total_aa.csv', index=False)

####
#aa_features()