<h2>Machine Learning-based Directed Evolution </h2>

`Data`:
 - Protabank: https://www.protabank.org/study_analysis/WkyQPR9A/
     - Essentially, we have 193 unique sequences and ~7 different assays performed on each sequence
     - These assays include Expression, Total energy, Binding energy, etc.
     - *Expression* is the primary output that we try to predict

 - ChR: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005786
    - There are 218 ChR chimeras chosen from a library of ~120k variants
    - *Expression* and *localization* are the primary output we try to predict
    - measured by GFP and mKate Fluorescence
 
`Protabank`/`MLDE`/`DNA_BERT`:
 - `baselines`: contains custom shallow MLP models and standard architectures (TODO)
 - `Protabank Figures`: contains results of experiments
 - `model_suite_train.py`: runs model suite on dataset of interest
 - Apply "zhihan1996/DNABERT-2-117M" as a Transformer baseline; have backbones of LSTMs and MLPs