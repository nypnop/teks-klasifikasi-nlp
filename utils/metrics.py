from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def text_classification_metrics_fn(list_hyp, list_label):
    accuracy = accuracy_score(list_label, list_hyp)
    f1 = f1_score(list_label, list_hyp, average="macro")
    recall = recall_score(list_label, list_hyp, average="macro")
    precision = precision_score(list_label, list_hyp, average="macro")
    return {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision
    }