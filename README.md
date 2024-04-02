# DVMPDC
A deep learning model based on dual-view representation and multi-strategy pooling, named DVMPDC, is developed to predict synergistic drug combinations.

# Requirements
python == 3.7.12

pytorch == 1.9.0

Numpy == 1.21.6

scikit-learn == 1.0.2

PyTorch Geometry == 2.0.3

# Files
### 1.data
label_oneil_loewe.csv: The benchmark dataset.We utilized the synergy dataset compiled in the study by Preuer et al. (Preuer K, Lewis R P I, Hochreiter S, et al. DeepSynergy: predicting anti-cancer drug synergy with Deep Learning. Bioinformatics, 2018, 34(9): 1538-1546). A total of 22,737 samples were obtained. After preprocessing, the benchmark dataset we constructed comprises 6,618 positive samples and 7,373 negative samples, involving 38 drugs and 33 cell lines.

oneil_cell.csv：Cell line feature matrix. Gene expression data of cell lines were obtained from the CCLE database (Barretina J, Caponigro G, Stransky N, et al. The Cancer Cell Line Encyclopedia enables predictive modeling of anticancer drug sensitivity[J]. Nature, 2012, 483(7391): 603- 607.).

oneil_drug_two_smiles.csv：Drug feature matrix. The SMILES of the drug were obtained from the PubChem database (Kim S, Chen J, Cheng T, et al. PubChem 2023 update[J]. Nucleic acids research, 2023, 51(D1): D1373-D1380.).

### 2.code
data_preprocessing.py：Corresponding drug and cell line signatures are generated for each sample.

cv_train.py：This function can test the predictive performance of our model under five-fold cross-validation.

models.py：This function contains the network framework of our entire model.

layers.py：This function contains the implementation of the various layers used to build the model.

# Require input files
 You should prepare one synergy dataset matrix, one cell line feature matrix, and one drug feature matrix. All matrices are stored in CSV file format.

label_oneil_loewe:Synergy dataset matrix. Each row of the matrix contains two drugs, a cell line, a synergy score, and a classification label.

oneil_cell.csv:Cell line feature matrix. Each row contains a cell line and the gene expression level of that cell line.

oneil_drug_two_smiles:Drug feature matrix. Each row contains a drug and the SMILES for that drug.

# Train and test folds
python cv_train.py

# Contact
If you have any questions or suggestions with the code, please let us know. Contact Chenliang Xie at xiechenliang@csu.edu.cn
