"""
Пользовательские функции вычисления метрик классификации (numpy).
Подключается к ноутбуку: import metrics_cls as cm
"""

import numpy as np


def _prepare_arrays(y_true, y_pred):
    return np.asarray(y_true), np.asarray(y_pred)


def accuracy(y_true, y_pred):
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    return np.mean(y_true == y_pred)


def confusion_matrix_np(y_true, y_pred, labels=None):
    y_true, y_pred = _prepare_arrays(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[label_to_idx[true_label], label_to_idx[pred_label]] += 1

    return matrix


def precision_macro(y_true, y_pred, labels=None):
    cm = confusion_matrix_np(y_true, y_pred, labels=labels)
    precisions = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        denom = tp + fp
        precisions.append(0.0 if denom == 0 else tp / denom)
    return float(np.mean(precisions))


def recall_macro(y_true, y_pred, labels=None):
    cm = confusion_matrix_np(y_true, y_pred, labels=labels)
    recalls = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        denom = tp + fn
        recalls.append(0.0 if denom == 0 else tp / denom)
    return float(np.mean(recalls))


def f1_macro(y_true, y_pred, labels=None):
    cm = confusion_matrix_np(y_true, y_pred, labels=labels)
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def all_metrics(y_true, y_pred, labels=None):
    return {
        "Accuracy": round(accuracy(y_true, y_pred), 4),
        "Precision": round(precision_macro(y_true, y_pred, labels=labels), 4),
        "Recall": round(recall_macro(y_true, y_pred, labels=labels), 4),
        "F1": round(f1_macro(y_true, y_pred, labels=labels), 4),
    }
