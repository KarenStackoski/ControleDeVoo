import pandas as pd

df = pd.read_csv('Data/flights.csv', low_memory=False)

# Cria a coluna alvo
df['ATRASOU'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# Seleciona colunas relevantes
colunas = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'SCHEDULED_DEPARTURE', 
           'DISTANCE', 'DEPARTURE_DELAY', 'ATRASOU']
df = df[colunas].dropna()

# Salva uma amostra de 10.000 linhas para o Orange
df.sample(n=10000, random_state=42).to_csv('Data/voos_orange.csv', index=False)
print("Arquivo salvo!")