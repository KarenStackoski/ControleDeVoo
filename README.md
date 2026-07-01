# Predição de Atrasos em Voos ✈️

Projeto da disciplina de Inteligência Artificial — Machine Learning para predição de atrasos em voos com Python.

---

## Tecnologias utilizadas

- Python 3.11+
- pandas
- scikit-learn
- XGBoost
- matplotlib
- seaborn

---

## Como rodar o projeto

### 1. Clonar o repositório

```bash
git clone https://github.com/KarenStackoski/ControleDeVoo.git
cd ControleDeVoo
```

### 2. Instalar as bibliotecas

```bash
python -m pip install pandas scikit-learn xgboost matplotlib seaborn
```

### 3. Baixar o dataset

O dataset não está incluído no repositório por ser grande demais (565MB).

1. Acesse: https://www.kaggle.com/datasets/usdot/flight-delays
2. Faça login ou crie uma conta gratuita no Kaggle
3. Clique em **Download**
4. Extraia o arquivo zip
5. Copie o arquivo `flights.csv` para dentro da pasta `Data/` do projeto

A estrutura deve ficar assim:
```
ControleDeVoo/
├── Data/
│   └── flights.csv   ← coloque o arquivo aqui
├── main.py
├── preparar_dados.py
└── ...
```

### 4. Rodar o modelo principal

```bash
python main.py
```

O script vai:
- Carregar e preparar os dados
- Treinar os modelos Random Forest e XGBoost
- Exibir acurácia, recall e relatório completo no terminal
- Salvar os gráficos (curva ROC, matrizes de confusão) em `resultados.png`

### 5. (Opcional) Gerar amostra reduzida

Caso queira gerar uma amostra menor do dataset (usada para testes):

```bash
python preparar_dados.py
```

Isso cria o arquivo `Data/voos_orange.csv` com 10.000 linhas.

---

## Resultados obtidos

| Modelo | Acurácia | Recall | Precision |
|--------|----------|--------|-----------|
| Random Forest | 93,64% | 71,30% | 91% |
| XGBoost | 94,01% | 74,45% | 90% |

Os gráficos completos estão disponíveis no arquivo `resultados.png`.

---

## Grupo

Trabalho desenvolvido em grupo para a disciplina de Inteligência Artificial.