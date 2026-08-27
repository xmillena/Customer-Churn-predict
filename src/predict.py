#%%

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

latest_version = max(
    [
        int(mv.version)
        for mv in client.search_model_versions("name='model-churn'")
    ]
)

#importar modelo
model_uri = f"models:/model-churn/{latest_version}"
model = mlflow.sklearn.load_model(model_uri)

#%%
features = model.feature_names_in_
features
#%%
amostra = pd.read_csv("../data/ultimas_10_linhas.csv")
predicao = model.predict_proba(amostra[features])[:,1]
amostra['proba'] = predicao
amostra[['proba']]
# %%

def categorizar_acao(row):
    if row['proba'] >= 0.70:
        return 'Risco alto'
    elif row['proba'] >= 0.40:
        return 'Risco médio'
    else:
        return 'Risco baixo'

# %%
amostra['nivel_risco'] = amostra.apply(categorizar_acao, axis=1)
amostra[['proba', 'nivel_risco']]
# %%
