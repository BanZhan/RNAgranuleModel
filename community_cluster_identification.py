import numpy as np
import pandas as pd

score_threshold = 0.5

network_d = pd.read_csv('9606.protein.physical.links.full.v12.0.txt', sep=' ')
print(network_d.columns, network_d.shape)
network = network_d[(network_d['experiments'] > 0)][['protein1', 'protein2']].drop_duplicates().values
print(network.shape)

protein_list = pd.read_csv('interactome_results.csv')
print('overall proteins: ', protein_list.shape)
protein_all = protein_list[protein_list['ave_proba_PBSG'] >= score_threshold]


# select the community for each location
# , 'nucleus', 'endoplasmic reticulum', 'cytosol'

GOCC = 'all_' # for the overall community'''

print('first selected identified RNA granule proteome: ', protein_all.shape)
print('selected protein id: ', protein_all.columns)

#GOCC = 'endoplasmic reticulum' # , 'nucleus', 'endoplasmic reticulum', 'cytosol'
#protein_all = protein_all[protein_all[GOCC] == 1]


print('second selected identified RNA granule proteome: ', protein_all.shape)
print('selected protein id: ', protein_all.columns)
protein_target = protein_all['#string_protein_id'].values

all_features = ['#string_protein_id', 'protein_name', 'length', 'IEP', 'aromaticity', 'entropy',
       'cation_frac', 'molecular_weight', 'gravy', 'alpha_helix', 'beta_turn',
       'beta_sheet', 'hpi_<-1.5_frac', 'hpi_<-2.0_frac', 'hpi_<-2.5_frac',
       'hpi_<-1.5', 'hpi_<-2.0', 'hpi_<-2.5', 'lcs_scores', 'ave_proba_PBSG']
node_classes = protein_all[all_features].values

protein_all[all_features].to_csv('node_classes_save.csv')

selected_interac_0 = {}

for p1, p2 in network:
	if (p1 in protein_target) or (p2 in protein_target):
		if p1 not in protein_target:
			if p1 not in selected_interac_0:
				selected_interac_0[p1] = 0
			selected_interac_0[p1] += 1
		elif p2 not in protein_target:
			if p2 not in selected_interac_0:
				selected_interac_0[p2] = 0
			selected_interac_0[p2] += 1


inter_list = pd.DataFrame([selected_interac_0]).T
inter_list.columns = ['interac_with_RNAgpro_num']
#inter_list.to_csv('interac_p_'+str(score_threshold)+'.csv')



#####################################################################################
#### identified RNA granule protein community building
import numpy as np
import pandas as pd
import os
import math
import warnings
from scipy import io
import scipy.stats as stats
from datetime import datetime
import random
import re
import shutil

import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')


def edge_id_gene_id(node_classes):
    save_id = {}
    save_attr = []
    save_id_proba = {}
    for i in range(len(node_classes)):
        save_id[node_classes[i, 0]] = i
        save_id_proba[node_classes[i, 0]] = node_classes[i, -1]
        save_attr.append(node_classes[i][1:])
    return save_id, save_attr, save_id_proba


save_id, save_attr, save_id_proba = edge_id_gene_id(node_classes)
#print('save id: ', save_id)

def select_edges(all_inter, save_id, save_id_proba):
    edge_save = []
    edge_weight = []
    for inter1, inter2 in all_inter:
        if inter1 in save_id and inter2 in save_id:
            id_1 = save_id[inter1]
            id_2 = save_id[inter2]
            edge_save.append([id_1, id_2])

            proba_1 = save_id_proba[inter1]
            proba_2 = save_id_proba[inter2]
            edge_weight.append(proba_1*proba_2)
    return edge_save, edge_weight

#
edge_save, edge_weight = select_edges(network, save_id, save_id_proba)

#print('edge_save: ', edge_save[0:100])
#print('edge_weight: ', edge_weight[0:100])

all_features = ['#string_protein_id', 'protein_name', 'length', 'IEP', 'aromaticity', 
'entropy', 'cation_frac', 'molecular_weight', 'gravy', 'alpha_helix', 'beta_turn',
       'beta_sheet', 'hpi_<-1.5_frac', 'hpi_<-2.0_frac', 'hpi_<-2.5_frac',
       'hpi_<-1.5', 'hpi_<-2.0', 'hpi_<-2.5', 'lcs_scores', 'ave_proba_PBSG']

def generate_graph(edge_save, save_id, save_attr):
    G = nx.Graph()

    for i in range(len(save_id)):
        #print(save_attr[i][0], save_attr[i][1], save_attr[i][2], save_attr[i][3])
        node_data = {'protein_name': save_attr[i][0], 
        'length':save_attr[i][1], 
        'IEP':save_attr[i][2],
        'aromaticity':save_attr[i][3],
        'entropy':save_attr[i][4],
        'cation_frac':save_attr[i][5],
        'molecular_weight':save_attr[i][6],
        'gravy':save_attr[i][7],
        'alpha_helix':save_attr[i][8],
        'beta_turn':save_attr[i][9],
        'beta_sheet':save_attr[i][10],
        'hpi_<-1.5_frac':save_attr[i][11],
        'hpi_<-2.0_frac':save_attr[i][12],
        'hpi_<-2.5_frac':save_attr[i][13],
        'hpi_<-1.5':save_attr[i][14],
        'hpi_<-2.0':save_attr[i][15],
        'hpi_<-2.5':save_attr[i][16],
        'lcs_scores':save_attr[i][17],
        'ave_proba_PBSG':save_attr[i][18],
        }
        G.add_node(i, **node_data)

    for e in edge_save:
        G.add_edge(e[0], e[1])
    return G

G = generate_graph(edge_save, save_id, save_attr)



def generate_weight_graph(edge_save, save_id, save_attr, edge_weight):
    G = nx.Graph()

    for i in range(len(save_id)):
        #print(save_attr[i][0], save_attr[i][1], save_attr[i][2], save_attr[i][3])
        node_data = {'protein_name': save_attr[i][0], 
        'length':save_attr[i][1], 
        'IEP':save_attr[i][2],
        'aromaticity':save_attr[i][3],
        'entropy':save_attr[i][4],
        'cation_frac':save_attr[i][5],
        'molecular_weight':save_attr[i][6],
        'gravy':save_attr[i][7],
        'alpha_helix':save_attr[i][8],
        'beta_turn':save_attr[i][9],
        'beta_sheet':save_attr[i][10],
        'hpi_<-1.5_frac':save_attr[i][11],
        'hpi_<-2.0_frac':save_attr[i][12],
        'hpi_<-2.5_frac':save_attr[i][13],
        'hpi_<-1.5':save_attr[i][14],
        'hpi_<-2.0':save_attr[i][15],
        'hpi_<-2.5':save_attr[i][16],
        'lcs_scores':save_attr[i][17],
        'ave_proba_PBSG':save_attr[i][18],
        }
        G.add_node(i, **node_data)

    for j in range(len(edge_save)):
        e = edge_save[j]
        w = edge_weight[j]
        G.add_edge(e[0], e[1], weight = w)
    return G

G_weight = generate_weight_graph(edge_save, save_id, save_attr, edge_weight)

# clustering
# Louvain Community
comp_non_louvain_com = nx.community.louvain_communities(G, weight = None, resolution=1.2, seed=123)
print(comp_non_louvain_com)

comp_weight_louvain_com = nx.community.louvain_communities(G_weight, weight = 'weight', resolution=1.2, seed=123)
print(comp_weight_louvain_com)


####
#save the clustering files into a csv
import csv
import networkx as nx

# Assuming comp_non_louvain_com and comp_weight_louvain_com are lists of communities

# Function to save clustering results to CSV
def save_communities_to_csv(communities, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node', 'Community'])
        for idx, community in enumerate(communities):
            for node in community:
                writer.writerow([node, idx])

# Save non-weighted Louvain communities
save_communities_to_csv(comp_non_louvain_com, 'non_weighted_louvain_communities_re1.2.csv')

# Save weighted Louvain communities
save_communities_to_csv(comp_weight_louvain_com, 'weighted_louvain_communities_re1.2.csv')

'''
# girvan_newman
comp_girvan_newman = nx.community.girvan_newman(G)
tuple(sorted(c) for c in next(comp_girvan_newman))'''