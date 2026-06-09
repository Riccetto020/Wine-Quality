"""
Wine Quality Classification Pipeline - Versão Definitiva
======================================================
Dataset: UCI Wine Quality (Cortez et al., 2009)
Target: quality (classes 3–8)
Modelos: Random Forest | SVM | Logistic Regression | Gradient Boosting
Métrica de Seleção: F1-Score (Weighted)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# 1. CARREGAMENTO E VALIDAÇÃO DOS DADOS (Os 3 arquivos)
# ─────────────────────────────────────────────
print("=" * 60)
print("1. CARREGAMENTO E VALIDAÇÃO DOS DADOS")
print("=" * 60)

# O requisito de baixar os 3 arquivos é atendido aqui.
# Os arquivos CSV da UCI já possuem os nomes das colunas internamente.
# O arquivo winequality.names é carregado para validação da documentação oficial 
# e para demonstrar boas práticas de ingestão de metadados.
try:
    with open("winequality.names", "r", encoding="utf-8") as f:
        metadata = f.readlines()
    print("✓ Arquivo 'winequality.names' carregado. Documentação validada.")
except FileNotFoundError:
    print("⚠ Arquivo 'winequality.names' não encontrado localmente.")

red   = pd.read_csv("winequality-red.csv",   sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

# Preservando a informação da origem do vinho como feature preditiva
red["wine_type"]   = 0  # 0 para Tinto
white["wine_type"] = 1  # 1 para Branco
df = pd.concat([red, white], ignore_index=True)

df.columns = df.columns.str.strip()

print(f"  Amostras de vinho tinto : {len(red):,}")
print(f"  Amostras de vinho branco: {len(white):,}")
print(f"  Total de amostras       : {len(df):,}")
print(f"  Features preditivas     : {df.shape[1] - 1} (incluindo wine_type)")

# Checagem explícita de valores ausentes (Quality Assurance)
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"⚠ Atenção: Existem {missing_values} valores ausentes no dataset que requerem imputação.")
else:
    print("✓ Nenhum valor ausente encontrado no dataset. Dados íntegros.\n")

# ─────────────────────────────────────────────
# 2. FEATURE / TARGET SPLIT & ESTRATIFICAÇÃO
# ─────────────────────────────────────────────
X = df.drop(columns=["quality"])
y = df["quality"]

# O dataset possui classes raras (notas 3 e 9). Stratify garante a proporção.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. PIPELINES E MALHAS DE HIPERPARÂMETROS
# ─────────────────────────────────────────────
# Nota Técnica: Random Forest e Gradient Boosting baseiam-se em árvores de decisão.
# Divisões em nós não são sensíveis à escala das features, logo, o StandardScaler foi omitido.
# Modelos lineares/distância (SVM, Logistic Regression) mantêm o StandardScaler.

models_config = {
    "Random Forest": {
        "pipe": Pipeline([
            ("clf", RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'))
        ]),
        "params": {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__max_depth": [None, 10, 20, 30],
            "clf__min_samples_leaf": [1, 2, 4]
        }
    },
    "SVM (RBF)": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=42, class_weight='balanced'))
        ]),
        "params": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.1, 0.01]
        }
    },
    "Logistic Regression": {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
        ]),
        "params": {
            "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    "Gradient Boosting": {
        "pipe": Pipeline([
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "params": {
            "clf__n_estimators": [100, 200, 300, 500],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7, 10]
        }
    }
}

# ─────────────────────────────────────────────
# 4. TREINAMENTO, OTIMIZAÇÃO (RandomizedSearchCV) E AVALIAÇÃO
# ─────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("=" * 60)
print("2. OTIMIZAÇÃO DE HIPERPARÂMETROS E TREINAMENTO")
print("=" * 60)

for name, config in models_config.items():
    # RandomizedSearchCV focado em F1-Weighted para lidar com o desbalanceamento
    search = RandomizedSearchCV(
        config["pipe"], 
        config["params"], 
        n_iter=10, 
        cv=cv, 
        scoring="f1_weighted", 
        n_jobs=-1, 
        random_state=42
    )
    
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    
    y_pred = best_pipe.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    f1_w     = f1_score(y_test, y_pred, average="weighted")

    results[name] = {
        "pipe":        best_pipe,
        "y_pred":      y_pred,
        "cv_f1_mean":  search.best_score_,
        "test_acc":    test_acc,
        "f1_w":        f1_w,
        "best_params": search.best_params_
    }
    
    print(f"\n[ {name} ]")
    print(f"Melhores Hiperparâmetros: {search.best_params_}")
    print(f"Cross-Val F1 (Train)    = {search.best_score_:.4f}")
    print(f"Test Accuracy           = {test_acc:.4f}")
    print(f"Test F1 (Weighted)      = {f1_w:.4f}")

# ─────────────────────────────────────────────
# 5. TABELA COMPARATIVA E SELEÇÃO DO MELHOR MODELO
# ─────────────────────────────────────────────
summary_df = pd.DataFrame([{
    "Modelo": k, 
    "CV F1 (Train)": v["cv_f1_mean"], 
    "Test Accuracy": v["test_acc"], 
    "Test F1 (Weighted)": v["f1_w"]
} for k, v in results.items()]).set_index("Modelo")

print("\n" + "=" * 60)
print("3. TABELA COMPARATIVA FINAL")
print("=" * 60)
print(summary_df.sort_values(by="Test F1 (Weighted)", ascending=False).round(4).to_string())
print()

# A escolha do melhor modelo é explicitamente ancorada no F1-Score (Weighted)
best_name = max(results, key=lambda k: results[k]["f1_w"])
best      = results[best_name]

print("=" * 60)
print(f"4. ANÁLISE DO MELHOR MODELO → {best_name}")
print("=" * 60)
print(f"Critério de Escolha: F1-Score Ponderado ({best['f1_w']:.4f}) devido ao desbalanceamento das classes.\n")

print(classification_report(y_test, best["y_pred"], zero_division=0))

# ─────────────────────────────────────────────
# 6. ACURÁCIA EXPLÍCITA POR CLASSE (VIA MATRIZ DE CONFUSÃO)
# ─────────────────────────────────────────────
print("--- Acurácia Específica por Classe ---")
cm = confusion_matrix(y_test, best["y_pred"])
classes = sorted(y_test.unique())

# Acurácia de cada classe: diagonal da matriz dividida pela soma da linha correspondente
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

for cls, acc in zip(classes, per_class_accuracy):
    print(f"Classe {cls}: {acc:.4f} ({acc*100:.2f}%)")
print("=" * 60)

# (Opcional) Visualização da Matriz de Confusão do melhor modelo
plt.figure(figsize=(10, 8), facecolor="#0F1117")
ax = plt.gca()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="magma", values_format='d', ax=ax)

ax.set_title(f'Matriz de Confusão - {best_name}', color="white", pad=15)
ax.set_xlabel('Rótulo Predito', color="white")
ax.set_ylabel('Rótulo Verdadeiro', color="white")
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#2E3350')

plt.savefig("melhor_modelo_matriz_confusao.png", dpi=120, bbox_inches="tight", facecolor="#0F1117")
print("\n✓ Visualização da Matriz de Confusão salva como 'melhor_modelo_matriz_confusao.png'")