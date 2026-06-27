# Projeto IA — Predição de Atrasos em Voos
## Resumo do que foi feito + Tutorial Orange

---

# PARTE 1 — O QUE JÁ FIZEMOS

## Ambiente configurado
- **Python 3.14.3** instalado e funcionando
- **VS Code** instalado com extensão Python
- **Bibliotecas instaladas:** pandas, scikit-learn, xgboost, matplotlib, seaborn, jupyter

## Dataset
- **Nome:** 2015 Flight Delays and Cancellations
- **Fonte:** [Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Arquivo:** `flights.csv` (~565MB)
- O arquivo fica na pasta `Data/` local — **não sobe pro GitHub** por ser grande demais

## Repositório GitHub
- **URL:** https://github.com/KarenStackoski/ControleDeVoo
- A pasta `Data/` está no `.gitignore` — cada membro baixa o dataset separado
- Para clonar o projeto:
```
git clone https://github.com/KarenStackoski/ControleDeVoo.git
```

## Arquivos do projeto
```
ControleDeVoo/
├── Data/                  ← pasta local, não vai pro GitHub
│   ├── flights.csv        ← dataset original (baixar do Kaggle)
│   └── voos_orange.csv    ← amostra gerada pelo preparar_dados.py
├── main.py                ← código principal do modelo
├── preparar_dados.py      ← gera o voos_orange.csv para o Orange
├── resultados.png         ← gráficos gerados pelo modelo
├── README.md              ← instruções do projeto
└── .gitignore             ← ignora a pasta Data/
```

---

## Modelos treinados

Dois modelos de machine learning foram treinados para prever se um voo vai atrasar (sim/não):

### Random Forest
| Métrica | Resultado |
|---------|-----------|
| Acurácia | 93,64% |
| Recall | 71,30% |
| Precision | 91% |

### XGBoost
| Métrica | Resultado |
|---------|-----------|
| Acurácia | 94,01% |
| Recall | 74,45% |
| Precision | 90% |

**Conclusão:** O XGBoost foi levemente superior. Ambos os modelos apresentaram ótimo desempenho.

---

## O que cada métrica significa

- **Acurácia:** de todos os voos previstos, quantos o modelo acertou. XGBoost acertou 94 em cada 100.
- **Recall:** dos voos que realmente atrasaram, quantos o modelo conseguiu identificar. XGBoost identificou 74 em cada 100 atrasos reais.
- **Precision:** quando o modelo disse "vai atrasar", quantas vezes estava certo. 90% das vezes estava certo.
- **Curva ROC / AUC:** mede o quanto o modelo consegue separar as duas classes (atrasou vs não atrasou). Quanto mais perto de 1.0, melhor.

---

## Como rodar o projeto do zero

**1. Clone o repositório:**
```
git clone https://github.com/KarenStackoski/ControleDeVoo.git
cd ControleDeVoo
```

**2. Instale as bibliotecas:**
```
pip install pandas scikit-learn xgboost matplotlib seaborn
```

**3. Baixe o dataset:**
- Acesse https://www.kaggle.com/datasets/usdot/flight-delays
- Baixe e extraia o zip
- Crie a pasta `Data/` e coloque o `flights.csv` dentro

**4. Gere o arquivo para o Orange:**
```
python preparar_dados.py
```

**5. Rode o modelo principal:**
```
python main.py
```

Os gráficos serão salvos em `resultados.png`.

---

# PARTE 2 — TUTORIAL ORANGE

## O que é o Orange?
O Orange é uma ferramenta visual de machine learning — você monta o modelo arrastando blocos, sem precisar escrever código. É ótimo para apresentações.

## Instalação
1. Acesse **orangedatamining.com**
2. Clique em **Download** e baixe para Windows
3. Instale normalmente (Next em tudo)

---

## Montando o fluxo no Orange

### Passo 1 — Carregar os dados
> ⚠️ Use o arquivo `voos_orange.csv` (gerado pelo `preparar_dados.py`), não o `flights.csv` original — ele é grande demais e trava o Orange.

- Na aba **Data** à esquerda, arraste o bloco **CSV File Import** para a área central
- Clique duas vezes nele e selecione o arquivo `Data/voos_orange.csv`
- Feche a janelinha que abrir

### Passo 2 — Selecionar colunas
- Na aba **Data**, arraste o bloco **Select Columns** e conecte ao **CSV File Import**
- Clique duas vezes nele e configure:
  - **Target (alvo):** `ATRASOU`
  - **Features (entradas):** `MONTH`, `DAY_OF_WEEK`, `AIRLINE`, `SCHEDULED_DEPARTURE`, `DISTANCE`, `DEPARTURE_DELAY`
  - **Ignored (ignorar):** qualquer outra coluna
- Feche a janela

### Passo 3 — Adicionar o modelo
- Na aba **Model**, arraste o bloco **Random Forest** para a área central
- Não precisa configurar nada, deixa os valores padrão

### Passo 4 — Avaliar o modelo
- Na aba **Evaluate**, arraste o bloco **Test and Score**
- Conecte assim:
  - **Select Columns** → **Test and Score**
  - **Random Forest** → **Test and Score**
- Clique duas vezes no **Test and Score** para ver os resultados

### Como deve ficar o fluxo completo:
```
CSV File Import → Select Columns → Test and Score
                                 ↑
                           Random Forest
```

### Passo 5 — Ver a Curva ROC
- Na aba **Evaluate**, arraste o bloco **ROC Analysis**
- Conecte o **Test and Score** ao **ROC Analysis**
- Clique duas vezes para ver o gráfico da curva ROC

---

## Dicas importantes
- Sempre use o `voos_orange.csv` (10.000 linhas) — o dataset completo trava o Orange
- Se o Orange travar, feche pelo Gerenciador de Tarefas (Ctrl+Alt+Delete) e reabra
- Os resultados no Orange podem ser levemente diferentes do Python — é normal, pois usa uma amostra menor

---

## Resumo para a apresentação

> *"Utilizamos Python com scikit-learn e XGBoost para treinar dois modelos de classificação, atingindo 94% de acurácia e 74% de recall. O modelo consegue prever corretamente a grande maioria dos atrasos com base em atributos como companhia aérea, horário de partida e distância do voo. Também visualizamos o fluxo do modelo na ferramenta Orange Data Mining."*
