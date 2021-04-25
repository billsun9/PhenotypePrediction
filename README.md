# Phenotype Prediction


## .FNA Data Loading
see seq_data_processing/

###### includes...
- [ ] seqio loading (faster?)
- [x] nucleotide seq to tensor via 1-hot (use tf v instead of sklearn v)

###### todo...

## Knockout/Knockin Generation
see seq_generation/

###### includes...
- [x] boilerplate scrape code

###### todo...
- [ ] scrape [SynergyAge](http://synergistic.aging-research.group/roundworm/) for gene-lifespan change pairs
- [ ] process data? (Inconsistencies in temp; change in lifespan metrics; epistasis)
- [ ] make baseline feedforward nn