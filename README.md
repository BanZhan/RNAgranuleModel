# RNAgranuleModel
## Package installation (Python 3.8.8)
Biopython package (Version: 1.80);
Scipy package (Version 1.9.3);
scikit-learn package (Version 1.3.0);
dcor package (Version 0.6)

## Model training
This section provides step-by-step instructions to train the RNA granule prediction models.
### Prerequisites
1. Download the dataset 'RNAgranule_model_training_data.zip' from the ResearchGate (https://www.researchgate.net/publication/386099818_Training_data_for_the_RNA_granule_model).
2. Clone or download all files from the 'model training' directory in this repository.
### Setup
3. Extract the dataset and place all files (from both the dataset and repository) in a single directory. Your directory structure should look like:
project_directory/  
├── aa_features_pre.py  
├── data_processing.py  
├── fasta_transfer.py  
├── IDR_identify.py  
├── model_train.py  
├── my_model_SG_1.pickle - my_model_SG_10.pickle  
├── pdb30.csv  
├── physicochemical_property_cal.py  
├── RNA_granule_2Kmer_tier1_result.csv  
├── RNA_granule_3Kmer_tier1_result.csv  
├── RNA_granule_data.xlsx  
├── SG_auc_df.csv  
├── total_aa.csv  
├── total_data.csv  
├── total_data.fasta  
├── uniprot_human_proteome.csv
### Training Models
4. Run the model training script with the appropriate parameters:
For P-body/Stress granule model: Set target = 'PBSG' in Line 189 and tier = 1 in Line 1
For P-body model only: Set target = 'PB' in Line 189 and tier = 2 in Line 1
Execute the training by running:
python model_train.py

## Predicting
The dataset for predicting can be downloaded in https://www.researchgate.net/publication/386100715_RNAgranule_model_predicting_data.
The codes in the 'Predicting' are used for predicting new proteins.
1. To predict the human proteome
   You can run the 'model_predict_proteome.py' file.
2. To predict your own proteins
   You can replace the information of each column in the 'uniprot_human_proteome.csv' file.

## Functional dense PPI cluster analysis
You can find the whole RNA granule proteome PPI community in the SI.xlsx file.
To visualize the identified functional dense PPI clusters:
1. you can download the SI.xlsx file
2. collect the cluster 1, cluster 2 and cluster 3 protein (prob>0.5) lists, OR cluster 1-, cluster 2- and cluster 3- protein (prob≥0.7) lists in each sheet
3. input the protein names of the 'N_1' column in the STRING website (Version: 12.0, link: https://version-12-0.string-db.org/).
4. use default basic settings to visualize and analyze your proteins of interest and see the interactions.
------------------------------------------------------------------------------
Feel free to send me an email if you need any help about the codes.
The files are under the CC BY-NC license.
