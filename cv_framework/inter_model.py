from typing import Dict, List, Tuple
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def calculate_inter_model_metrics(results: List[Dict]) -> Tuple[np.ndarray, List[str], dict, np.ndarray]:
    """Calcula a matriz de acordo, teste de McNemar e falhas genuínas sem gerar gráficos."""
    if len(results) < 2:
        return np.array([]), [], {}, np.array([])

    preds_dict = {res["Modelo"]: np.array(res["test_preds"]) for res in results}
    true_labels = np.array(results[0]["test_labels"])
    model_names = list(preds_dict.keys())

    # 1. Matriz de Acordo
    agreement = np.zeros((len(model_names), len(model_names)))
    for i, mod1 in enumerate(model_names):
        for j, mod2 in enumerate(model_names):
            agreement[i, j] = np.mean(preds_dict[mod1] == preds_dict[mod2])

    # 2. McNemar
    top_model = max(results, key=lambda x: x["Test_F1-Macro"])["Modelo"]
    mcnemar_results = {}
    p1 = preds_dict[top_model]
    
    for mod2 in model_names:
        if mod2 == top_model: continue
        p2 = preds_dict[mod2]
        contingency_table = [
            [np.sum((p1 == true_labels) & (p2 == true_labels)), np.sum((p1 == true_labels) & (p2 != true_labels))],
            [np.sum((p1 != true_labels) & (p2 == true_labels)), np.sum((p1 != true_labels) & (p2 != true_labels))]
        ]
        mcnemar_results[f"{top_model}_vs_{mod2}"] = mcnemar(contingency_table, exact=True).pvalue

    # 3. Falhas Genuínas
    all_preds = np.array([preds_dict[m] for m in model_names])
    erros_totais_idx = np.where(np.sum(all_preds == true_labels, axis=0) == 0)[0]

    return agreement, model_names, mcnemar_results, erros_totais_idx