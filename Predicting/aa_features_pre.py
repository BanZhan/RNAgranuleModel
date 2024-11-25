# inspired by the PSAP model
# https://github.com/vanheeringen-lab/psap/blob/master/psap/matrix.py
#Publication | Mierlo, G., Jansen, J. R. G., Wang, J., Poser, I., van Heeringen, S. J., & Vermeulen, M. (2021). Predicting protein condensate formation using machine learning. Cell Reports, 34(5), 108705. https://doi.org/10.1016/j.celrep.2021.108705.

import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from tqdm.auto import tqdm
import time
from scipy import signal
from IDR_identify import fasta_prep_IDR, collect_IDR_scores
# Import collections module
import collections
# Import scipy.stats module
import scipy.stats as stats

RESIDUES = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

# Kyte & Doolittle {kd} index of hydrophobicity
HP = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "U": 0.0,
}

cov_window = 30

def fasta2df(dbfasta):
    """
    Read peptide sequences and attributes from fasta file and convert to Pandas DataFrame.
    """
    print("Converting peptide fasta to data frame")
    rows = list()
    with open(dbfasta) as f:
        for record in SeqIO.parse(f, "fasta"):
            seqdict = dict()
            seq = str(record.seq)
            id = record.description
            rows.append([id, seq])
    df = pd.DataFrame(rows, columns=["protein_name", "sequence"])
    return df

# Define a function to calculate entropy of a sequence based on Shannon's defination
def Shannon_entropy_cal(seq):
	# Count the frequency of each character using collections.Counter
	freq = collections.Counter(seq)

	# Calculate the probability of each character by dividing by the length of sequence
	prob = [f / len(seq) for f in freq.values()]

	# Calculate the entropy using scipy.stats.entropy function
	ent = stats.entropy(prob, base=2)
	# Return the entropy value
	return ent

def cation_fraction_cal(seq):
	# define the cationic amino acids
	cationic_aa = ['K', 'R', 'H']

	# calculate the total number of cationic amino acids in the protein sequence
	num_cationic = sum(seq.count(aa) for aa in cationic_aa)

	# calculate the total number of amino acids in the protein sequence
	num_aa = len(seq)

	# calculate the cation fraction of the protein sequence
	cation_fraction = num_cationic / num_aa
	#print("Cation fraction of the protein sequence:", cation_fraction)

	return cation_fraction

def IDR_cal(seq):
	#
	fasta_prep_IDR([seq])
	IDR_results = collect_IDR_scores()
	IDR_seq = ''.join([seq[i] for i in range(len(IDR_results)) if IDR_results[i] == 1])
	return round(len(IDR_seq)/len(seq), 5)

def add_lowcomplexityscore(df):
    """
    Add lowcomplexity score to data frame.
    """
    print("Adding lowcomplexity score to data frame")
    lcs_window = 20
    lcs_cutoff = 7
    for index, row in df.iterrows():
        seq = str(row["sequence"])
        if len(seq) > lcs_window + 1:
            sig = list()
            for i in range(len(seq)):
                window = seq[i : i + lcs_window]
                if len(window) == lcs_window:
                    acid_comp = len(list(set(window)))
                    sig.append(acid_comp)
            score = sum([1 if i <= 7 else 0 for i in sig])
            df.loc[index, "lcs_score"] = score
            df.loc[index, "lcs_fraction"] = score / len(sig)

def add_lowcomplexity_features(df):
    """
    Adds lowcomplexity features to data frame.
    """
    print("Adding lowcomplexity features")

    n_window = 20
    cutoff = 7
    n_halfwindow = int(n_window / 2)
    lcs_lowest_complexity = list()
    lcs_scores = list()
    lcs_fractions = list()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # for index, row in self.df.iterrows():
        # Determine low complexity scores
        seq = str(row["sequence"])
        lcs_acids = list()
        sig = list()
        # New
        lc_bool = [False] * len(seq)
        for i in range(len(seq)):
            if i < n_halfwindow:
                peptide = seq[:n_window]
            elif i + n_halfwindow > int(len(seq)):
                peptide = seq[-n_window:]
            else:
                peptide = seq[i - n_halfwindow : i + n_halfwindow]
            complexity = len(set(peptide))
            if complexity <= 7:
                for bool_index in (i - n_halfwindow, i + n_halfwindow):
                    try:
                        lc_bool[bool_index] = True
                    except IndexError:
                        pass
                lcs_acids.append(seq[i])
            sig.append(complexity)
        # Adding low complexity scores to list
        low_complexity_list = pd.DataFrame(
            {"bool": lc_bool, "acid": list(seq)}, index=None
        )
        lcs_lowest_complexity.append(min(sig))
        lcs_scores.append(
            len(low_complexity_list.loc[low_complexity_list["bool"] == True])
        )
        lcs_fractions.append(
            len(low_complexity_list.loc[low_complexity_list["bool"] == True])
            / len(seq)
        )
        low_complexity_list = pd.DataFrame(
            {"bool": lc_bool, "acid": list(seq)}, index=None
        )
        # Add default values
        for i in RESIDUES:
            df.loc[index, i + "_lcscore"] = 0
            df.loc[index, i + "_lcfraction"] = 0
        if len(lcs_acids) >= n_window:
            for i in RESIDUES:
                df.loc[index, i + "_lcscore"] = len(
                    low_complexity_list.loc[
                        (low_complexity_list["bool"] == True)
                        & (low_complexity_list["acid"] == i)
                    ]
                )
                df.loc[index, i + "_lcfraction"] = len(
                    low_complexity_list.loc[
                        (low_complexity_list["bool"] == True)
                        & (low_complexity_list["acid"] == i)
                    ]
                ) / len(lcs_acids)
    df["lcs_fractions"] = lcs_fractions
    df["lcs_scores"] = lcs_scores
    df["lcs_lowest_complexity"] = lcs_lowest_complexity
    return df

class HydroPhobicIndex:
    def __init__(self, hpilist):
        self.hpilist = hpilist

def hydrophobic(df):
    for index, row in df.iterrows():
        hpilst = pd.Series(list(row["sequence"])).map(HP).tolist()
        df.loc[index, "HydroPhobicIndex"] = HydroPhobicIndex(hpilst)
    return df

def add_hydrophobic_features(df):
    print("Adding hydrophobic features")
    hpi0, hpi1, hpi2, hpi3, hpi4, hpi5 = (
        list(),
        list(),
        list(),
        list(),
        list(),
        list(),
    )
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # for index, row in self.df.iterrows():
        # Convolve signal
        win = signal.hann(cov_window)
        sw = signal.convolve(
            row["HydroPhobicIndex"].hpilist, win, mode="same"
        ) / sum(win)
        # Append features
        hpi0.append(sum(i < -1.5 for i in sw) / len(sw))
        # self.df.loc[index, 'hpi_<-1.5_frac'] = hpi
        hpi1.append(sum(i < -2.0 for i in sw) / len(sw))
        # self.df.loc[index, 'hpi_<-2.0_frac'] = hpi
        hpi2.append(sum(i < -2.5 for i in sw) / len(sw))
        # self.df.loc[index, 'hpi_<-2.5_frac'] = hpi
        hpi3.append(sum(i < -1.5 for i in sw))
        # self.df.loc[index, 'hpi_<-1.5'] = hpi
        hpi4.append(sum(i < -2.0 for i in sw))
        # self.df.loc[index, 'hpi_<-2.0'] = hpi
        hpi5.append(sum(i < -2.5 for i in sw))
        # self.df.loc[index, 'hpi_<-2.5'] = hpi
    df["hpi_<-1.5_frac"] = hpi0
    df["hpi_<-2.0_frac"] = hpi1
    df["hpi_<-2.5_frac"] = hpi2
    df["hpi_<-1.5"] = hpi3
    df["hpi_<-2.0"] = hpi4
    df["hpi_<-2.5"] = hpi5
    return df


def One_amino_acid_analysis(df):
    """
    fraction of amino acid residues (defined in RESIDUES) to data frame.
    """
    print("One amino acid fractions")
    for res in RESIDUES:
        df["fraction_" + res] = (
            df["sequence"].str.count(res) / df["sequence"].str.len()
        )
    df["length"] = df["sequence"].str.len()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # for index, row in self.df.iterrows():
        seq = row["sequence"]
        seqanalysis = ProteinAnalysis(seq)
        acidist = seqanalysis.get_amino_acids_percent()
        df.loc[index, "IEP"] = seqanalysis.isoelectric_point()
        # Calculates the aromaticity value of a protein according to Lobry & Gautier (1994, Nucleic Acids Res., 22, 3174-3180).
        #It is simply the relative frequency of Phe+Trp+Tyr (i.e., F, W, Y).
        df.loc[index, "aromaticity"] = seqanalysis.aromaticity()
        # Shannon entropy calculation
        df.loc[index, "entropy"] = Shannon_entropy_cal(seq)
        # cation fraction using 'K', 'R', 'H'
        df.loc[index, "cation_frac"] = cation_fraction_cal(seq)
        # IDR fraction
        #df.loc[index, "IDR_frac"] = IDR_cal(seq)
        df.loc[index, "molecular_weight"] = seqanalysis.molecular_weight()
        df.loc[index, "gravy"] = seqanalysis.gravy()

        # add_biochemical_combinations
        df = df.assign(
            alpha_helix=df["fraction_V"]
            + df["fraction_I"]
            + df["fraction_Y"]
            + df["fraction_F"]
            + df["fraction_W"]
            + df["fraction_L"]
        )
        df = df.assign(
            beta_turn=df["fraction_N"]
            + df["fraction_P"]
            + df["fraction_G"]
            + df["fraction_S"]
        )
        df = df.assign(
            beta_sheet=df["fraction_E"]
            + df["fraction_M"]
            + df["fraction_A"]
            + df["fraction_L"]
        )

    df = hydrophobic(df)
    print(df)
    df = add_hydrophobic_features(df)
    return df


def kmer_count(all_kmers, seqs, k, protein_names):
    result = pd.DataFrame(index=all_kmers)
    for i, sequence in enumerate(seqs):
        kmers = [sequence[j:j+k] for j in range(len(sequence)-k+1)]
        kmers_perc = pd.Series(kmers).value_counts()/len(sequence)
        result[i] = kmers_perc
    result = result.fillna(0).astype(float)
    result.index.name = 'kmer'
    result.reset_index(inplace=True)
    return result.T.values


def prep_select_kmer(df0, df, num_kmer, k):
    # select top kmers for further analysis
    select_kmers = df0.sort_values(by='kmer_occurence', ascending=False)
    select_kmers = select_kmers[select_kmers['p_values_vs_proteome'] <= 0.001]
    select_kmers = select_kmers.head(num_kmer).values[:, 0]
    print('select_kmers: ', select_kmers)
    kmer_get = kmer_count(select_kmers, df['sequence'].values, k, df['protein_name'].values)
    kmer_df = pd.DataFrame(kmer_get[1:, :], columns = kmer_get[0, :])
    df = pd.concat([df, kmer_df], axis=1)
    return df


def kmer_fraction_lcs(df):
    original_df_2 = pd.read_csv('RNA_granule_2Kmer_tier1_result.csv')
    original_df_3 = pd.read_csv('RNA_granule_3Kmer_tier1_result.csv')
    #original_df_4 = pd.read_csv('RNA_granule_4Kmer_tier1_result.csv')

    num_kmer_2 = 50; k = 2
    df_2 = prep_select_kmer(original_df_2, df, num_kmer_2, k)

    num_kmer_3 = 50; k = 3
    df_3 = prep_select_kmer(original_df_3, df_2, num_kmer_3, k)

    #num_kmer_4 = 250; k = 4
    #df_4 = prep_select_kmer(original_df_4, df_3, num_kmer_4, k)
    #print('kmer 1, 2, 3, 4(', num_kmer_4)
    return df_3


def aa_features_final(fasta_file):
	df_data = fasta2df(fasta_file)
	#print(df_data.shape)
	df_1_0 = One_amino_acid_analysis(df_data)
	#print(df_1_0.shape)
	df_1_1 = add_lowcomplexity_features(df_1_0).drop(['HydroPhobicIndex'], axis=1)
	#print(df_1_1.shape)
	total_1_2_3aa = kmer_fraction_lcs(df_1_1)
	#print(total_1_2_3aa.shape)
	return total_1_2_3aa



######
# example
'''
df_1aa_2aa_3aa = aa_features_final('new.fasta')
df_1aa_2aa_3aa.to_csv('pdb30_check2.csv', index=False)'''