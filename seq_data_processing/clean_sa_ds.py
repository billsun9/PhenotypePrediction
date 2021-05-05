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