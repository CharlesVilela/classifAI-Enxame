import pandas as pd
import os
from os.path import join

from pathlib import Path
base_path = Path(__file__).resolve().parents[2]

# Lista global de resultados
results = []

def log_result(algorithm, model_name, best_params, metrics, exec_time):
    """
    Armazena os resultados de cada experimento na lista global.
    """

    if os.path.exists(join(base_path, "log", "log.xlsx")):
        results = pd.read_csv(join(base_path, "log", "log.xlsx"))

    results.append({
        "Algorithm": algorithm,
        "Model": model_name,
        "Best Params": str(best_params),
        "Accuracy": metrics.get("accuracy", None),
        "Precision": metrics.get("precision", None),
        "Recall": metrics.get("recall", None),
        "F1-score": metrics.get("f1", None),
        "Exec Time (s)": exec_time
    })

    if not results:
        print("⚠️ Nenhum resultado para salvar.")
        return
    
    results.to_csv(join(base_path, "log", "log.csv"),sep=";", index=False)

    print(f"✅ Resultado logado: {algorithm} | {model_name}")

def save_results_to_csv(path="output/optimization_results.csv"):
    """
    Salva os resultados em CSV.
    """
    if not results:
        print("⚠️ Nenhum resultado para salvar.")
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    print(f"💾 Resultados salvos em {path}")
