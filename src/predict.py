"""

import pandas as pd

model_df = pd.read_pickle("model.pkl")
model = model_df['model']
features = model_df['features']

#%%
#depois separar os dados la pra poder simular uma prediçao pra suar aqui
df = pd.read_csv(r"../data/raw/telco_info.csv")
amostra = df[df['dtRef'] == df['dtRef'].max()].sample(3)

# %%
predicao = model.predict_proba(amostra[features])[:,1]
#%%

amostra['proba'] = predicao"""