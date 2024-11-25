# Import scipy.stats module
import scipy.stats as stats
# Import collections module
import collections

from Bio.SeqUtils import ProtParam, ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from IDR_identify import fasta_prep_IDR, collect_IDR_scores


def protein_property_cal(seq, IDR_c):
	seq = seq.replace('X', '')
	if len(seq) < 1:
		return round(-99, 2), round(-99, 2), round(-99, 4), round(-99, 4), round(-99, 4), round(-99, 4), round(-99, 4), round(-99, 4)
	# create a ProteinAnalysis object
	protein = ProteinAnalysis(seq)
	# calculate the molecular weight
	mw = protein.molecular_weight()
	#print("Molecular weight of the protein sequence:", round(mw, 2))

	# calculate the isoelectric point
	pi = protein.isoelectric_point()

	# calculate the instability index
	ii = protein.instability_index()

	#Calculate the hydrophilicity using analysis.gravy method
	gravy = protein.gravy()

	# Calculates the aromaticity value of a protein according to Lobry & Gautier (1994, Nucleic Acids Res., 22, 3174-3180).
	#It is simply the relative frequency of Phe+Trp+Tyr (i.e., F, W, Y).
	aromaticity = protein.aromaticity()

	## define the Kyte and Doolittle hydrophobicity scale
	#kd_hydrophobicity = ProtParamData.kd
	# calculate the Kyte and Doolittle hydrophobicity index for the protein sequence
	#kd_hydrophob = protein.protein_scale(kd_hydrophobicity, 5, edge=1)
	# calculate the average Kyte and Doolittle hydrophobicity score for the protein sequence
	#ave_kd_hydrophob = sum(kd_hydrophob) / len(kd_hydrophob)

	# Shannon entropy calculation
	entropy = Shannon_entropy_cal(seq)

	# cation fraction using 'K', 'R', 'H'
	cation_frac = cation_fraction_cal(seq)

	if not IDR_c:
		return round(mw, 2), round(pi, 2), round(ii, 4), round(gravy, 4), round(entropy, 4), round(aromaticity, 4), round(cation_frac, 4), round(1, 4)
	# IDR fraction
	IDR_frac = IDR_cal(seq)

	#print(mw, pi, gravy, ave_kd_hydrophob, entropy, aromaticity, cation_frac, IDR_frac)
	return round(mw, 2), round(pi, 2), round(ii, 4), round(gravy, 4), round(entropy, 4), round(aromaticity, 4), round(cation_frac, 4), round(IDR_frac, 4)


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

####################
#  main function
####################
def engineered_properties_cal(seqs, IDR_c):
	print('wight properties: Molecular_Weight, Isoelectric_Point, instability_index, Hydrophilicity, Shannon_Entropy, Aromatic_Fraction, Cation_Fraction, IDR_Fraction')

	engineered_properties = []
	for seq in seqs:
		Molecular_Weight, Isoelectric_Point, instability_index, Hydrophilicity, Shannon_Entropy, Aromatic_Fraction, Cation_Fraction, IDR_Fraction = protein_property_cal(seq, IDR_c)
		engineered_properties.append([Molecular_Weight, Isoelectric_Point, instability_index, Hydrophilicity, Shannon_Entropy, Aromatic_Fraction, Cation_Fraction, IDR_Fraction])

	fasta_prep_IDR(seqs)

	return engineered_properties


'''
# define your protein sequence
protein_seq = ["LADHYG"]
eight_properties = engineered_properties_cal(protein_seq)
print(eight_properties)

# prepare fasta sequences for IDR analysis
'''