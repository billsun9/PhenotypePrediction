# Phenotype Prediction


## .FNA Data Loading
see seq_data_processing/

###### includes...
- [ ] seqio loading (faster?)
- [x] nucleotide seq (ATCG) to tensor via one-hot (use tf v instead of sklearn v)
- [x] gene seq to 1-d vec (use tf v)
- [x] cleaned synergyage data --> `./data/SynergyAge_DS/Cleaned_SynergyAge_Database.csv`

###### todo...

## Knockout/Knockin Generation
see seq_generation/
training scripts in `./seq_generation/train`; WIP
###### includes...
- [x] Janggu lib testing --> kind of buggy and sometimes deprecated; dont recommend

###### todo...
- [ ] make baseline feedforward nn
- [ ] explore CGR cnn-based model
