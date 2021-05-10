# converts raw synergy_age data into nicer csv
import pandas as pd

path = "../data/SynergyAge_DS/SynergyAge_Database.csv"

raw_ds = pd.read_csv(path)
# %%
raw_ds.columns
raw_ds = raw_ds.drop(
    ['Pubmed ID', 'Details','Intervention on gene 1','Intervention(s) on gene 2(,3)','Phenotype description','Organism']
            ,axis=1)

# %%
for name in raw_ds.columns:
    print(raw_ds[name].describe())
    print("isnull--->"+str(raw_ds[name].isnull().sum()))
    print('---')
# %%
# %%
# must compare vs wildtype; so drop all rows w/o wildtype metric
filtered_ds = raw_ds[raw_ds['Wild type lifespan (days)'].notna()]
# %%
print(raw_ds.isnull().sum())


# %%
tmp = filtered_ds[['Wild type lifespan (days)', 'Gene 1','Gene 1 single mutant lifespan (days)']]
tmp = tmp.dropna()
# %%
tmp2 = filtered_ds[['Wild type lifespan (days)','Gene(s) 2(,3)','Gene(s) 2(,3) mutant lifespan (days)']]
tmp2.dropna()
# %%
tmp3 = filtered_ds[['Wild type lifespan (days)','Gene 1','Gene(s) 2(,3)','Double (triple) mutant (genes 1,2,(3)) lifespan (days)']]
tmp3.dropna()
tmp3['All genes'] = tmp3[['Gene 1','Gene(s) 2(,3)']].agg(';'.join, axis=1)
# %%
cols_for3 = list(tmp3.columns.values)
cols_for3 = ['Wild type lifespan (days)','All genes','Double (triple) mutant (genes 1,2,(3)) lifespan (days)']
# %%
tmp3 = tmp3[cols_for3]
# %%
for m in [tmp,tmp2,tmp3]:
    m.columns=['Wild Lifespan', 'Gene(s)', 'Mutant Lifespan']
clean_ds=pd.concat([tmp,tmp2,tmp3],axis=0, ignore_index=True)
# %%
clean_ds.to_csv("../data/SynergyAge_DS/Cleaned_SynergyAge_Database.csv",index=False)


# %%
clean_ds['Gene(s)'].describe()
clean_ds['Gene(s)'].value_counts()
clean_ds['Wild Lifespan'].describe()
clean_ds['Mutant Lifespan'].describe()
# %%
gene_1_intervention = raw_ds['Intervention on gene 1']
gene_1_intervention_counts = gene_1_intervention.value_counts()

out_path_1 = '../data/SynergyAge_DS/gene_1_counts.csv'
gene_1_intervention_counts.to_csv(out_path_1)

# %%
gene_2_3_intervention = raw_ds['Intervention(s) on gene 2(,3)']
gene_2_3_intervention_counts = gene_2_3_intervention.value_counts()

out_path_2 = '../data/SynergyAge_DS/gene_2_3_counts.csv'
gene_2_3_intervention_counts.to_csv(out_path_2)
# %%
import re
def interventions_clean_and_to_list(path, col_name):
    clean_sa_ds = pd.read_csv(path)
    l = []
    for cur_genes in list(set(list(clean_sa_ds[str(col_name)]))):
        cur_genes = cur_genes.replace(" ","")
        genes = re.split(";|,",cur_genes)
        for gene in genes:
            if gene not in l:
                l.append(gene)
    return l


gene_1_list = interventions_clean_and_to_list(path, col_name='Intervention on gene 1')
gene_2_3_list = interventions_clean_and_to_list(path, col_name='Intervention(s) on gene 2(,3)')

import pickle
out_path_list_1 = 'data_analytics/intervention_gene_1_unique.pkl'
out_path_list_2_3 = 'data_analytics/intervention_gene_2_3_unique.pkl'

with open(out_path_list_1, 'wb') as f:
    pickle.dump(gene_1_list, f)

with open(out_path_list_2_3, 'wb') as f:
    pickle.dump(gene_2_3_list, f)


# to load...
# import pickle
# with open(out_path_list_1, 'rb') as f:
#     gene_1_unique_interventions_list = pickle.load(f)














