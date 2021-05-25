# Phenotype Prediction


## Data Loading/Processing
see seq_data_processing/

### includes...
- [x] nucleotide seq (ATCG) to tensor via one-hot (use tf v instead of sklearn v)
- [x] gene seq to 1-d vec (use tf v)
- [x] cleaned synergyage data --> `./data/SynergyAge_DS/sa_clean_data.csv`
- [ ] seqio loading (faster?)

### todo...

## Knockout/Knockin Generation
see seq_generation/

training scripts in `./seq_generation/train`

utility functions (gene seq to 1-d vec; 1-d vec to gene seq; rmse; mae) in `./seq_generation/utils`
### includes... 
- [x] Initial janggu lib testing --> kind of buggy and sometimes deprecated; dont recommend
- [x] optimized xgb model --> `./seq_generation/train/models/xgb.pickle`
- [x] nn model --> `./seq_generation/train/nn_dense/`
- [x] xgb model + nn predictions on entire dataset; gridsearch performance --> `./seq_generation/train/model_results/`
### todo...
- [ ] Janggu + Geneius libs
- [ ] explore CGR cnn-based model
- [ ] DNABert 