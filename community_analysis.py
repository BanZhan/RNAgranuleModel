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
'''
GOCC = 'endoplasmic reticulum' # , 'nucleus', 'endoplasmic reticulum', 'cytosol'
protein_all = protein_all[protein_all[GOCC] == 1]
'''

print('second selected identified RNA granule proteome: ', protein_all.shape)
print('selected protein id: ', protein_all.columns)
protein_target = protein_all['#string_protein_id'].values

all_features = ['#string_protein_id', 'protein_name', 'length', 'IEP', 'aromaticity', 'entropy',
       'cation_frac', 'molecular_weight', 'gravy', 'alpha_helix', 'beta_turn',
       'beta_sheet', 'hpi_<-1.5_frac', 'hpi_<-2.0_frac', 'hpi_<-2.5_frac',
       'hpi_<-1.5', 'hpi_<-2.0', 'hpi_<-2.5', 'lcs_scores', 'ave_proba_PBSG']
node_classes = protein_all[all_features].values

'''
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
inter_list.to_csv('interac_p_'+str(score_threshold)+'.csv')
'''



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
    for i in range(len(node_classes)):
        save_id[node_classes[i, 0]] = i
        save_attr.append(node_classes[i][1:])
    return save_id, save_attr


save_id, save_attr = edge_id_gene_id(node_classes)
#print('save id: ', save_id)

def select_edges(all_inter, save_id):
    edge_save = []
    for inter1, inter2 in all_inter:
        if inter1 in save_id and inter2 in save_id:
            id_1 = save_id[inter1]
            id_2 = save_id[inter2]
            edge_save.append([id_1, id_2])
    return edge_save

#
edge_save = select_edges(network, save_id)


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
# write the network to a GEXF file
nx.write_gexf(G, GOCC+"RNAg_com_network"+str(score_threshold)+".gexf")


# TSNE for 2d distribution of nodes
# Perform t-SNE dimensionality reduction on the graph
# Convert the NodeView object to a NumPy array
# Extract the unique proteins from the list of interactions
proteins = save_id
print('proteins: ', proteins)
# Create an empty matrix with dimensions equal to the number of unique proteins
matrix = np.zeros((len(proteins), len(proteins)))

# Fill in the matrix with the interaction data
for inter1, inter2 in network:
    #print('interaction: ', inter1, inter2)
    if inter1 in save_id and inter2 in save_id:
        i = proteins[inter1]
        j = proteins[inter2]
        matrix[i, j] = 1
        matrix[j, i] = 1

print('matrix: ', matrix)

node_array = np.array(matrix)
tsne = TSNE(n_components=2, random_state=1)
node_embeddings = tsne.fit_transform(node_array)


df = pd.DataFrame()
df["proteins"] = list(proteins)
df["comp-1"] = node_embeddings[:,0]
df["comp-2"] = node_embeddings[:,1]

df.to_csv(GOCC+'TSNE_community_0411_1.csv')





# Compute node importance
# Extract degree and weighted degree
degree = G.degree(weight=None)
weighted_degree = G.degree(weight='weight')

# Convert to DataFrame
degree_centrality = nx.degree_centrality(G)

# calculate betweenness centrality
bet_cen = nx.betweenness_centrality(G, seed=1)

# calculate eigenvector centrality
eig_cen = nx.eigenvector_centrality(G)

# calculate PageRank
page_rank = nx.pagerank(G)

# calculate closeness centrality
clo_cen = nx.closeness_centrality(G)

# calculate clustering coefficient
clus_cof = nx.clustering(G)

# Store node importance in a data frame
df = pd.DataFrame({
    'protein_target': list(protein_target),
    'degree': list(degree),
    'weighted_degree': list(weighted_degree),
    'bet_cen': list(bet_cen.values()),
    'eig_cen': list(eig_cen.values()),
    'page_rank': list(page_rank.values()),
    'clo_cen': list(clo_cen.values()),
    'clus_cof': list(clus_cof.values()),
    'Degree Centrality': list(degree_centrality.values())
})
df.to_csv(GOCC+'RNAg_com_network_node_features'+str(score_threshold)+'0411_1.csv')






##################################################################
# selected node ids in the network
'''
def find_protein_class(selected_protein_id, all_inter, node_classes):
    protein_class_dict = {node: 1 if node in selected_protein_id else 2 for node in node_classes[:,0]}
    #print(protein_class_dict)
    for inter1, inter2 in all_inter:
        #print(inter1, inter2)
        if (inter1 in protein_class_dict) & (inter2 in protein_class_dict):
            if protein_class_dict[inter1] + protein_class_dict[inter2] == 3:
                if protein_class_dict[inter1] == 1:
                    protein_class_dict[inter2] = 3
                else:
                    protein_class_dict[inter1] = 3
    protein_class = [[node, protein_class_dict[node]] for node in node_classes[:,0]]
    return protein_class



target_granule = 'PB_predicted' # 'SG_predicted', 'PB/SG_predicted'
selected_protein_if_df = prediction_df[prediction_df[target_granule] == 5]
selected_protein_id = selected_protein_if_df['Stable Gene ID'].values
protein_class_results = find_protein_class(selected_protein_id, all_inter.values, node_classes)
df[['Stable Gene ID_pb','Protein Class_pb']] = protein_class_results


target_granule = 'SG_predicted' # 'SG_predicted', 'PB/SG_predicted'
selected_protein_if_df = prediction_df[prediction_df[target_granule] == 5]
selected_protein_id = selected_protein_if_df['Stable Gene ID'].values
protein_class_results = find_protein_class(selected_protein_id, all_inter.values, node_classes)
df[['Stable Gene ID_sg','Protein Class_sg']] = protein_class_results

target_granule = 'PB/SG_predicted' # 'SG_predicted', 'PB/SG_predicted'
selected_protein_if_df = prediction_df[prediction_df[target_granule] == 5]
selected_protein_id = selected_protein_if_df['Stable Gene ID'].values
protein_class_results = find_protein_class(selected_protein_id, all_inter.values, node_classes)
df[['Stable Gene ID_pb_sg','Protein Class_pb_sg']] = protein_class_results

df.to_csv(GOCC+'protein_class_in_the_graph.csv')
'''
