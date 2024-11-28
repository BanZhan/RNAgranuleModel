## RNAgranuleModel
# Package installation (Python 3.8.8)
Biopython package (Version: 1.80)
Scipy package (Version 1.9.3)
scikit-learn package (Version 1.3.0)
dcor package (Version 0.6)

# Model training
The dataset for model training can be downloaded in https://www.researchgate.net/publication/386099818_Training_data_for_the_RNA_granule_model.
The codes in the 'model training' are used for building RNA granule (i.e., P-body, stress granule, and P-body/stress granule) protein classification models, respectively.
You can run the model_train.py file to train the RNA granule models.

# Predicting
The dataset for predicting can be downloaded in https://www.researchgate.net/publication/386100715_RNAgranule_model_predicting_data.
The codes in the 'Predicting' are used for predicting new proteins.
1. To predict the human proteome
   You can run the 'model_predict_proteome.py' file.
2. To predict your own proteins
   You can replace the information of each column in the 'uniprot_human_proteome.csv' file.

# Functional dense PPI cluster analysis
You can find the whole RNA granule proteome PPI community in the SI.xlsx file.

------------------------------------------------------------------------------
Feel free to send me an email if you need any help about the codes.
The files are under the CC BY-NC license.
