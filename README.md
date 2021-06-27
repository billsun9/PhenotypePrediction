<h2 align="center">Phenotype Prediction</h2>


### Data
see data/

#### includes
 - [x] `./c_elegans_full_genome/` --> Full nucleotide sequence formatted as .txt files
 - [x] `./SynergyAge_DS/genes/` --> Cleaned SynergyAge DS for gene-%change lifespan pairs (initial approach)
 - [x] `./SynergyAge_DS/interventions/` --> Cleaned SynergyAge DS for intervention-%change lifespan pairs (current approach)
 
### Data Loading/Processing
see seq_data_processing/

#### includes
 - [x] `./data_analytics/` --> value counts for genes, collective interventions, individual interventions
 - [x] `./gene_processing/clean_sa_ds.py` --> script to clean sa dataset into gene-%change lifespan pairs
 - [x] `./nucleotide_processing/create_sa_intervention_db.py` --> script to clean sa dataset into intervention-lifespan pairs
 - [x] `./nucleotide_processing/wormbase_api_querying.py` --> script to query wormbase api for nucleotide data in json format

#### todo
 - [x] Load changes from wormbase API into .fasta files for c_elegans_full_genome
 - [ ] Data loading pipeline for ML models
 
### Knockout/Knockin Generation
see seq_generation/

naive gene-level training scripts in `./train/gene_level/`

utility functions (gene seq to 1-d vec; 1-d vec to gene seq; rmse; mae) in `./seq_generation/utils`
#### includes
 - [x] Gene level models
   - [x] `./models/gene_level/` --> xgb model, nn model, optimized xgb model and optimized nn model via gridsearching
   - [x] `./model_results/gene_level/` --> predictions on entire dataset, gridsearch performance
   - [x] `./inference/gene_level/` --> inference scripts
   
 - [ ] Nucleotide level models
   - [x] `./models/nucleotide_level/` --> initial nn model
   
#### todo
 - [ ] Janggu + Geneius libs
 - [ ] explore CGR cnn-based model
 - [ ] DNABert 