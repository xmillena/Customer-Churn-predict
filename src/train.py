
#%%
import numpy as np 
import pandas as pd 
#%%

df_data = pd.read_csv(r"..\data\raw\telco_info.csv")

df_data.head()

#%%

df_data.info()

#%%
#Preparando os dados
df = df_data.copy()

df = df.dropna(subset=['TotalCharges'])

genero = {'Female':1, 'Male':0}
binario = {'Yes':1, 'No':0}

colunas_binarias = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
for col in colunas_binarias:
    df[col] = df[col].map(binario)

df['gender'] = df['gender'].map(genero)

cols_one_hot = ['InternetService', 'Contract', 'PaymentMethod']
df_final = pd.get_dummies(df, columns=cols_one_hot, drop_first = True)
bool_cols = df_final.select_dtypes(include=['bool']).columns
df_final[bool_cols] = df_final[bool_cols].astype(int)

cols_texto = [
    'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
]
for col in cols_texto:
    df_final[col] = df_final[col].replace({'No internet service': 'No', 'No phone service': 'No'})

    df_final[col] = df_final[col].map({'Yes': 1, 'No': 0})


df_final.head()

#removendo ultimas 10 linhas pra testar as previsoes depois
ultimas_linhas_pred = df_final.tail(10).drop(columns='Churn').copy()
ultimas_linhas_pred.to_csv('ultimas_10_linhas.csv', index=False)

df_final = df_final.iloc[:-10]

# Verifica o novo tamanho do df_data e do arquivo de teste
print(f"Total original: {len(df_final) + 10} linhas")
print(f"Linhas para treino: {df_final.shape[0]}")
print(f"Linhas para previsão: {ultimas_linhas_pred.shape[0]}")

#%%

list(df_final.columns)

#%%

#Seleção de features e separação dos dados
features = df_final.drop(columns=['customerID', 'Churn', 'TotalCharges']).columns.copy()
target = 'Churn'
X, y = df_final[features], df_final[target]

from sklearn import model_selection
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
# %%

df_analise = X_train.copy()
df_analise[target] = y_train
sumario = df_analise.groupby(by=target).agg(['mean', "median"]).T

sumario['diff_abs'] = sumario[0]-sumario[1]
sumario['diff_rel'] = sumario[0]/sumario[1] 

sumario.sort_values(by='diff_rel', ascending=False)
# %%

from sklearn import pipeline, tree

arvore = tree.DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)

arvore.feature_importances_
# %%
feature_importance = pd.Series(arvore.feature_importances_, index=X_train.columns).sort_values(ascending=False).reset_index()
feature_importance['acumulada'] = feature_importance[0].cumsum()
feature_importance[feature_importance[0]>0.01]
# %%
best_features = (feature_importance[feature_importance['acumulada']<0.96]['index'].tolist())
best_features
# %%
X_train.isnull().sum()
#%%


"Modelo Regressao linear"
#testar outros modelos
from sklearn import linear_model, ensemble

#model = linear_model.LogisticRegression(penalty=None, random_state=42, max_iter=1000)
model = ensemble.RandomForestClassifier(random_state=42, n_jobs=2)
params = {
    "min_samples_leaf":[15,20,25,30,50],
    "n_estimators": [100,200,300,1000],
    "criterion": ['gini', 'entropy', 'log_loss']   

}
grid = model_selection.GridSearchCV(model, params, cv=3, scoring="roc_auc")

#conforme eu for adicionando coisa pra fazer, adicionar mais steps discretization, onehot, etc
model_pipeline = pipeline.Pipeline(steps = [('Grid', grid)])
#grid.fit(X_train[best_features], y_train)
#%%
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experiment_name='churn-predict-exp')
with mlflow.start_run():
    mlflow.sklearn.autolog()
    model_pipeline.fit(X_train[best_features], y_train)

    # %%
    from sklearn import metrics

    y_train_predict = model_pipeline.predict(X_train[best_features])
    y_train_proba = model_pipeline.predict_proba(X_train[best_features])[:,1] 

    acc_train = metrics.accuracy_score(y_train, y_train_predict)
    auc_train = metrics.roc_auc_score(y_train, y_train_proba)

    print("Acurácia Treino: ", acc_train)
    print("AUC treino: ",auc_train)

    # %%

    y_test_predict = model_pipeline.predict(X_test[best_features])
    y_test_proba = model_pipeline.predict_proba(X_test[best_features])[:,1] 

    acc_test = metrics.accuracy_score(y_test, y_test_predict)
    auc_test = metrics.roc_auc_score(y_test, y_test_proba)

    print("Acurácia Test: ", acc_test)
    print("AUC test", auc_test)

    # %%

    model_df = pd.Series({
        'model':model_pipeline,
        'features': best_features,

    })

    model_df.to_pickle("model.pkl")
# %%

from mlflow import MlflowClient

client = MlflowClient()

for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)

    runs = client.search_runs([exp.experiment_id])
    print(f"Runs: {len(runs)}")
# %%
