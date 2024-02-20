<h2>Machine Learning-based Directed Evolution </h2>

`Data`:
 - https://www.protabank.org/study_analysis/WkyQPR9A/
     - Essentially, we have 193 unique sequences and ~7 different assays performed on each sequence
     - These assays include Expression, Total energy, Binding energy, etc.
     - *Expression* is the primary output that we try to predict
 
`Protabank`:
 - `baselines`: contains custom shallow MLP models and standard architectures (TODO)
 - `Protabank Figures`: contains results of experiments
 - `model_suite_train.py`: runs model suite on dataset of interest