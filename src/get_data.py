# %% IMPORTAR CSV PARA SQLITE
import sqlite3
import pandas as pd
import os

csv_path = r"data\raw\telco_info.csv" 

df = pd.read_csv(csv_path)

#conectar ao sqlite
conn = sqlite3.connect('churn_data.db')  

# salvar como tabela
df.to_sql('telco_raw', conn, if_exists='replace', index=False)

print(f" {len(df)} registros")
print(f"{len(df.columns)} colunas")
print(f"Tabela: telco_raw")

# verificar
query = "SELECT * FROM telco_raw LIMIT 5"
sample = pd.read_sql_query(query, conn)
print(f"Sample:")
print(sample)

conn.close()