# ============================================================
# 🔹 IMPORTS
# ============================================================
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import pyswarms as ps

# ============================================================
# 🔹 SEED GLOBAL PARA REPRODUTIBILIDADE
# ============================================================
random.seed(42)
np.random.seed(42)

# ============================================================
# 🔹 Função genérica de otimização PSO
# ============================================================
def pso_optimize(model_objective, bounds, dimensions, n_particles=20, iters=50):
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles,
        dimensions=dimensions,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=bounds
    )
    best_cost, best_pos = optimizer.optimize(model_objective, iters=iters)
    print(f"✅ Best cost: {best_cost}")
    return best_pos

# ============================================================
# 🔹 Funções objetivo com Cross-Validation e F1-Score Weighted
# ============================================================

def objective_lr_cv(hyperparams, X_train, y_train):
    results = []
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for params in hyperparams:
        C = np.clip(params[0], 0.01, 10)
        scores = []
        
        for train_idx, val_idx in skf.split(X_train_vec, y_train):
            X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = LogisticRegression(C=C, max_iter=500, solver='liblinear', random_state=42)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds, average='weighted')
            scores.append(f1)
        
        mean_f1 = np.mean(scores)
        results.append(1 - mean_f1)  # Minimizar

    return np.array(results)

def objective_rf_cv(hyperparams, X_train, y_train):
    results = []
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    for params in hyperparams:
        n_estimators = int(np.clip(params[0], 10, 200))
        max_depth = int(np.clip(params[1], 3, 30))
        min_samples_split = int(np.clip(params[2], 2, 10))
        
        scores = []
        
        for train_idx, val_idx in skf.split(X_train_vec, y_train):
            X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            f1 = f1_score(y_val, preds, average='weighted')
            scores.append(f1)
        
        mean_f1 = np.mean(scores)
        results.append(1 - mean_f1)  # Minimizar

    return np.array(results)

# ============================================================
# 🔹 Função de treino e avaliação final
# ============================================================
def train_and_evaluate(model, model_name, x_train, x_test, y_train, y_test):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', model)
    ])

    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    print(f"\n🔹 ### The Model: {model_name} ###")
    print(classification_report(y_test, predictions, digits=3))

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average="weighted"),
        "recall": recall_score(y_test, predictions, average="weighted"),
        "f1": f1_score(y_test, predictions, average="weighted")
    }
    print(f"🔧 Metrics: {metrics}")
    return predictions

# ============================================================
# 🔹 Pipeline principal integrado com PSO
# ============================================================
def build_pipeline_complete_pso(x, y):

    # Filtrar classes com pelo menos 2 amostras no dataset completo
    counts = y.value_counts()
    valid_classes = counts[counts >= 2].index
    x = x[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    counts_train_pso = y_train.value_counts()
    valid_classes_train_pso = counts_train_pso[counts_train_pso >= 2].index
    X_train = X_train[y_train.isin(valid_classes_train_pso)]
    y_train = y_train[y_train.isin(valid_classes_train_pso)]

    # # Dividir treino/validação para PSO
    # X_train_pso, X_val_pso, y_train_pso, y_val_pso = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    # )

    # ================= Naive Bayes =================
    nb_model = MultinomialNB()
    print("\n🔹 Treinando modelo: Naive Bayes")
    train_and_evaluate(nb_model, "Naive Bayes", X_train, X_test, y_train, y_test)

    # ================= Logistic Regression (PSO) =================
    bounds_lr = ([0.01], [10])
    best_pos_lr = pso_optimize(
        lambda x: objective_lr_cv(x, X_train, y_train),
        bounds=bounds_lr,
        dimensions=1,
        n_particles=20,
        iters=50
    )
    lr_model = LogisticRegression(C=best_pos_lr[0], max_iter=500, solver='liblinear', random_state=42)
    print(f"🔧 Melhores hiperparâmetros Logistic Regression: {best_pos_lr}")
    train_and_evaluate(lr_model, "Logistic Regression (PSO)", X_train, X_test, y_train, y_test)

    # ================= Random Forest (PSO) =================
    bounds_rf = ([10, 3, 2], [200, 30, 10])
    best_pos_rf = pso_optimize(
        lambda x: objective_rf_cv(x, X_train, y_train),
        bounds=bounds_rf,
        dimensions=3,
        n_particles=20,
        iters=50
    )
    rf_model = RandomForestClassifier(
        n_estimators=int(best_pos_rf[0]),
        max_depth=int(best_pos_rf[1]),
        min_samples_split=int(best_pos_rf[2]),
        random_state=42,
        n_jobs=-1
    )
    print(f"🔧 Melhores hiperparâmetros Random Forest: {best_pos_rf}")
    train_and_evaluate(rf_model, "Random Forest (PSO)", X_train, X_test, y_train, y_test)

# ============================================================
# 🔹 Exemplo de uso
# ============================================================
# df = pd.read_csv("seus_dados.csv")
# build_pipeline_complete_pso(df["text"], df["label"])
