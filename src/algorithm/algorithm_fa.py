# ============================================================
# 🔹 IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import random

from mealpy.swarm_based import FA
from mealpy import Problem, FloatVar, IntegerVar

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

# ============================================================
# 🔹 SEED GLOBAL
# ============================================================
random.seed(42)
np.random.seed(42)

# ============================================================
# 🔹 Função genérica de otimização FA
# ============================================================
def fa_optimize(model_objective, var_types, lb, ub, n_fireflies=20, iters=50):
    bounds = []
    for i, var_type in enumerate(var_types):
        if var_type == "float":
            bounds.append(FloatVar(lb[i], ub[i], name=f"x{i}"))
        elif var_type == "int":
            bounds.append(IntegerVar(lb[i], ub[i], name=f"x{i}"))

    problem = Problem(
        bounds=bounds,
        minmax="min",
        obj_func=model_objective,
        log_to="console"
    )

    model = FA.OriginalFA(epoch=iters, pop_size=n_fireflies)
    best_solution = model.solve(problem)
    print(f"✅ Best cost: {best_solution.target}")
    return best_solution.solution

# ============================================================
# 🔹 Função objetivo Logistic Regression com CV
# ============================================================
def objective_lr_cv(solution, X_train, y_train):
    C = np.clip(solution[0], 0.01, 10)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train_vec, y_train):
        X_tr, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = LogisticRegression(C=C, max_iter=500, solver='liblinear', random_state=42)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds, average='weighted')
        scores.append(f1)

    return 1 - np.mean(scores)

# ============================================================
# 🔹 Função objetivo Random Forest com CV
# ============================================================
def objective_rf_cv(solution, X_train, y_train):
    n_estimators = max(10, int(solution[0]))
    max_depth_value = solution[1]
    max_depth = None if max_depth_value <= 0 else int(max_depth_value)
    min_samples_split = max(2, int(solution[2]))

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
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

    return 1 - np.mean(scores)

# ============================================================
# 🔹 Função de treino e avaliação
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
# 🔹 Pipeline principal integrado com FA
# ============================================================
def build_pipeline_complete_fa(x, y):

    # Filtrar classes com >= 3 amostras
    counts_total = y.value_counts()
    valid_classes_total = counts_total[counts_total >= 3].index
    x = x[y.isin(valid_classes_total)]
    y = y[y.isin(valid_classes_total)]

    # Dividir treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    # Filtrar classes no treino com >=2 amostras
    counts_train = y_train.value_counts()
    valid_classes_train = counts_train[counts_train >= 2].index
    X_train = X_train[y_train.isin(valid_classes_train)]
    y_train = y_train[y_train.isin(valid_classes_train)]

    models = []

    # 🔹 Logistic Regression (FA)
    var_types_lr = ["float"]
    lb_lr = [0.01]
    ub_lr = [10.0]

    best_pos_lr = fa_optimize(
        lambda sol: objective_lr_cv(sol, X_train, y_train),
        var_types=var_types_lr,
        lb=lb_lr,
        ub=ub_lr,
        n_fireflies=20,
        iters=1
    )

    lr_model = LogisticRegression(C=best_pos_lr[0], max_iter=500, solver='liblinear', random_state=42)
    models.append((lr_model, "Logistic Regression (FA)", best_pos_lr))

    # 🔹 Random Forest (FA)
    var_types_rf = ["int", "float", "int"]
    lb_rf = [10, -10, 2]
    ub_rf = [200, 50, 20]

    best_pos_rf = fa_optimize(
        lambda sol: objective_rf_cv(sol, X_train, y_train),
        var_types=var_types_rf,
        lb=lb_rf,
        ub=ub_rf,
        n_fireflies=20,
        iters=1
    )

    n_estimators = max(10, int(best_pos_rf[0]))
    max_depth_value = best_pos_rf[1]
    max_depth = None if max_depth_value <= 0 else int(max_depth_value)
    min_samples_split = max(2, int(best_pos_rf[2]))

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    models.append((rf_model, "Random Forest (FA)", best_pos_rf))

    # 🔹 Treinar e avaliar todos os modelos
    for model, name, best_pos in models:
        print(f"\n🔧 Melhores hiperparâmetros para {name}: {best_pos}")
        train_and_evaluate(model, name, X_train, X_test, y_train, y_test)

# ============================================================
# 🔹 Exemplo de uso
# ============================================================
# df = pd.read_csv("seus_dados.csv")
# build_pipeline_complete_fa(df["text"], df["label"])
