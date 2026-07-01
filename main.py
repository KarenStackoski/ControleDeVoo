# =============================================================
# PROJETO IA para predição de atrasos em voos
# Bibliotecas: pandas, scikit-learn, xgboost, matplotlib
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("Carregando dataset...")
df = pd.read_csv('data/flights.csv', low_memory=False)
print(f"Dataset carregado! {df.shape[0]} linhas e {df.shape[1]} colunas.\n")


print("=== PRIMEIRAS LINHAS DO DATASET ===")
print(df.head())

print("\n=== COLUNAS DISPONÍVEIS ===")
print(df.columns.tolist())

print("\n=== INFORMAÇÕES GERAIS ===")
print(df.info())

print("\n=== VALORES NULOS POR COLUNA ===")
print(df.isnull().sum())

# Selecionando as colunas mais relevantes
colunas = [
    'MONTH', 'DAY_OF_WEEK', 'AIRLINE',
    'DEPARTURE_DELAY', 'ARRIVAL_DELAY',
    'SCHEDULED_DEPARTURE', 'DISTANCE',
    'AIR_TIME', 'DEPARTURE_TIME'
]

# Filtrando as colunas
colunas = [c for c in colunas if c in df.columns]
df = df[colunas].copy()

# Criação da coluna alvo 
# Regra: atrasou = 1 se atraso na chegada > 15 minutos
df['ATRASOU'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

print(f"\n=== DISTRIBUIÇÃO DA VARIÁVEL ALVO ===")
contagem = df['ATRASOU'].value_counts()
print(f"Não atrasou (0): {contagem[0]} voos")
print(f"Atrasou (1):     {contagem[1]} voos")

# Removendo linhas com valores nulos
df = df.dropna()
print(f"\nApós remover nulos: {df.shape[0]} linhas restantes.")

# Convertendo texto para número
df['AIRLINE'] = pd.Categorical(df['AIRLINE']).codes

# Removendo coluna de atraso na chegada
df = df.drop(columns=['ARRIVAL_DELAY'])

# Amostra de 100.000 linhas para rodar mais rápido
df = df.sample(n=100000, random_state=42)
print(f"Usando amostra de {len(df)} voos para treinamento.\n")

# Divisão: TREINO X TESTE

# X = entradas
# y = resultados
X = df.drop(columns=['ATRASOU'])
y = df['ATRASOU']

# 80% dos dados para treinar, 20% para testar
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Treino: {X_treino.shape[0]} voos | Teste: {X_teste.shape[0]} voos\n")

# ----- MODELO 1: Random Forest -----
print("Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_treino, y_treino)
y_pred_rf = rf.predict(X_teste)
y_prob_rf = rf.predict_proba(X_teste)[:, 1]
print("Random Forest treinado!\n")

# ----- MODELO 2: XGBoost -----
print("Treinando XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_treino, y_treino)
y_pred_xgb = xgb.predict(X_teste)
y_prob_xgb = xgb.predict_proba(X_teste)[:, 1]
print("XGBoost treinado!\n")

print("=" * 50)
print("RESULTADOS — RANDOM FOREST")
print("=" * 50)
print(f"Acurácia: {accuracy_score(y_teste, y_pred_rf):.2%}")
print(f"Recall:   {recall_score(y_teste, y_pred_rf):.2%}")
print("\nRelatório completo:")
print(classification_report(y_teste, y_pred_rf, target_names=['Não atrasou', 'Atrasou']))

print("=" * 50)
print("RESULTADOS — XGBOOST")
print("=" * 50)
print(f"Acurácia: {accuracy_score(y_teste, y_pred_xgb):.2%}")
print(f"Recall:   {recall_score(y_teste, y_pred_xgb):.2%}")
print("\nRelatório completo:")
print(classification_report(y_teste, y_pred_xgb, target_names=['Não atrasou', 'Atrasou']))


# ELABORAÇÃO DOS GRÁFICOS

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Predição de Atrasos em Voos — Resultados', fontsize=16, fontweight='bold')

# --- Gráfico 1: Distribuição de voos atrasados vs não atrasados ---
ax1 = axes[0, 0]
contagem.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax1.set_title('Distribuição: Atrasou vs Não Atrasou')
ax1.set_xlabel('0 = Não atrasou | 1 = Atrasou')
ax1.set_ylabel('Quantidade de voos')
ax1.set_xticklabels(['Não atrasou', 'Atrasou'], rotation=0)

# --- Gráfico 2: Matriz de Confusão — Random Forest ---
ax2 = axes[0, 1]
cm_rf = confusion_matrix(y_teste, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Não atrasou', 'Atrasou'],
            yticklabels=['Não atrasou', 'Atrasou'])
ax2.set_title('Matriz de Confusão — Random Forest')
ax2.set_ylabel('Real')
ax2.set_xlabel('Previsto')

# --- Gráfico 3: Matriz de Confusão — XGBoost ---
ax3 = axes[1, 0]
cm_xgb = confusion_matrix(y_teste, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=ax3,
            xticklabels=['Não atrasou', 'Atrasou'],
            yticklabels=['Não atrasou', 'Atrasou'])
ax3.set_title('Matriz de Confusão — XGBoost')
ax3.set_ylabel('Real')
ax3.set_xlabel('Previsto')

# --- Gráfico 4: Curva ROC — ambos os modelos ---
ax4 = axes[1, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_teste, y_prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_teste, y_prob_xgb)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

ax4.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', color='blue')
ax4.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})', color='orange')
ax4.plot([0, 1], [0, 1], 'k--', label='Aleatório (AUC = 0.500)')
ax4.set_title('Curva ROC')
ax4.set_xlabel('Taxa de Falsos Positivos')
ax4.set_ylabel('Taxa de Verdadeiros Positivos (Recall)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados.png', dpi=150, bbox_inches='tight')
print("\nGráficos salvos em 'resultados.png'!")
plt.show()

print("\nProjeto concluído com sucesso!")