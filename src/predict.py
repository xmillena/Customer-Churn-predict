#%%

import pandas as pd

model_df = pd.read_pickle("model.pkl")
model = model_df['model']
features = model_df['features']

#%%
"""amostra = pd.read_csv("../data/ultimas_10_linhas.csv")

# %%
predicao = model.predict_proba(amostra[features])[:,1]
#%%

amostra['proba'] = predicao
print(amostra[['proba'] + list(features)])"""
# %%
