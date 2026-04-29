import numpy as np                                                                                                    
import pandas as pd                                                                                                   
import optuna                                                                                                         
import matplotlib.pyplot as plt                                                                                       
import metrics_cls as cm
from sklearn.pipeline import Pipeline                     
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay)                                 
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold)  


SEARCH_CV = 3
KFOLD_SPLITS = 10
SCORING = 'f1_macro'
RANDOM_STATE = 42

# Экспоненциальное ядро для SVM: строит матрицу попарных сходств.
def exponential_kernel(gamma=0.1):
    def kernel(X1, X2):
        X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
        X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
        distances = np.sqrt(np.maximum(X1_sq + X2_sq - 2 * X1 @ X2.T, 0.0))
        return np.exp(-gamma * distances)
    return kernel


# Собирает pipeline модели; масштабирование включаем для методов, чувствительных к масштабу.
def make_pipeline(model, scale=True):
    if scale:
        return Pipeline([('scaler', StandardScaler()), ('model', model)])
    return Pipeline([('model', model)])


# Считает основные метрики классификации из sklearn с macro-усреднением.
def sklearn_metrics(y_true, y_pred):
    return {
        'F1': round(f1_score(y_true, y_pred, average='macro'), 4),
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4),
    }


# Сравнивает sklearn-метрики с пользовательской реализацией из metrics_cls.py.
def compare_metrics(y_true, y_pred, labels=None):
    sk = sklearn_metrics(y_true, y_pred)
    custom = cm.all_metrics(y_true, y_pred, labels=labels)
    row = {}
    for key, value in sk.items():
        row[f'sk {key}'] = value
    for key, value in custom.items():
        row[f'my {key}'] = value
    return row


# Обучает модель и сразу возвращает предсказания и метрики на train/test.
def evaluate_model(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    train_pred = model.predict(X_tr)
    test_pred = model.predict(X_te)
    train_metrics = sklearn_metrics(y_tr, train_pred)
    test_metrics = sklearn_metrics(y_te, test_pred)
    return model, train_pred, test_pred, train_metrics, test_metrics


# Выбирает лучший вариант модели по качеству на validation-выборке.
def choose_best_by_validation(candidates, X_tr, y_tr, X_val_local, y_val_local):
    fitted = {}
    scores = {}
    for name, estimator in candidates.items():
        estimator.fit(X_tr, y_tr)
        val_pred = estimator.predict(X_val_local)
        fitted[name] = estimator
        scores[name] = f1_score(y_val_local, val_pred, average='macro')
    best_name = max(scores, key=scores.get)
    return fitted[best_name], best_name, scores


# Запускает три способа подбора гиперпараметров: GridSearchCV, RandomizedSearchCV и Optuna.
def run_searches(base_pipeline, param_grid, param_distributions, optuna_builder, X_search, y_search, X_val, y_val, n_trials=10):
    grid = GridSearchCV(base_pipeline, param_grid, cv=SEARCH_CV, scoring=SCORING, n_jobs=1)
    grid.fit(X_search, y_search)

    rnd = RandomizedSearchCV(
        base_pipeline,
        param_distributions=param_distributions,
        n_iter=min(8, np.prod([len(v) for v in param_distributions.values()])),
        cv=SEARCH_CV,
        scoring=SCORING,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )
    rnd.fit(X_search, y_search)

    # Optuna оптимизирует качество модели на validation-выборке.
    def objective(trial):
        model = optuna_builder(trial)
        model.fit(X_search, y_search)
        pred = model.predict(X_val)
        return f1_score(y_val, pred, average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    opt_model = optuna_builder(study.best_trial)
    opt_model.fit(X_search, y_search)

    return {
        'GridSearchCV': grid.best_estimator_,
        'RandomizedSearchCV': rnd.best_estimator_,
        'Optuna': opt_model,
    }


# Считает метрики на 10 стратифицированных фолдах и сводит их в таблицу.
def cross_val_metrics(estimator_factory, X_data, y_data, balance_train=True, balance_fn=None):
    skf = StratifiedKFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data), start=1):
        X_tr = X_data.iloc[train_idx].reset_index(drop=True)
        y_tr = y_data.iloc[train_idx].reset_index(drop=True)
        X_te = X_data.iloc[test_idx].reset_index(drop=True)
        y_te = y_data.iloc[test_idx].reset_index(drop=True)

        # Балансировку проводим только внутри обучающей части каждого фолда.
        if balance_train and balance_fn is not None:
            X_tr, y_tr = balance_fn(X_tr, y_tr, random_state=RANDOM_STATE + fold)

        estimator = estimator_factory()
        estimator.fit(X_tr, y_tr)
        pred = estimator.predict(X_te)
        metrics = sklearn_metrics(y_te, pred)
        metrics['Fold'] = fold
        rows.append(metrics)

    fold_df = pd.DataFrame(rows)
    summary = {
        metric: f"{fold_df[metric].mean():.4f} ± {fold_df[metric].std():.4f}"
        for metric in ['F1', 'Accuracy', 'Precision', 'Recall']
    }
    return fold_df, summary


# Формирует одну строку итоговой hold-out таблицы по модели.
def holdout_row(name, train_metrics, test_metrics):
    return {
        'Классификатор': name,
        **{f'Train {k}': v for k, v in train_metrics.items()},
        **{f'Test {k}': v for k, v in test_metrics.items()},
    }


# Строит confusion matrix для выбранной модели на тестовой выборке.
def add_confusion_plot(model, X_te, y_te, title, labels=None):
    pred = model.predict(X_te)
    cmatrix = confusion_matrix(y_te, pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(title)
    plt.grid(False)
    plt.show()


# Confusion matrix (%) + bar chart реальных vs предсказанных классов.
# Вызывать после каждой модели для наглядной оценки качества.
def plot_model_results(model, X_te, y_te, title, labels=None):
    pred = model.predict(X_te)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13)

    # Нормализованная confusion matrix — показывает % ошибок по каждому классу
    cmatrix = confusion_matrix(y_te, pred, labels=labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cmatrix, display_labels=labels)
    disp.plot(ax=axes[0], cmap='Blues', values_format='.2f', colorbar=False)
    axes[0].set_title('Confusion Matrix (доля по строке)')
    axes[0].grid(False)

    # Bar chart: реальное распределение vs предсказанное
    label_list = labels if labels is not None else np.unique(np.concatenate([y_te, pred]))
    real_counts = [np.sum(np.asarray(y_te) == c) for c in label_list]
    pred_counts = [np.sum(pred == c) for c in label_list]
    x = np.arange(len(label_list))
    width = 0.35
    axes[1].bar(x - width / 2, real_counts, width, label='Реальные', color='#7aa6c2')
    axes[1].bar(x + width / 2, pred_counts, width, label='Предсказанные', color='#d9a46f')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_list)
    axes[1].set_xlabel('Класс')
    axes[1].set_ylabel('Количество')
    axes[1].set_title('Реальные vs Предсказанные')
    axes[1].legend()

    plt.tight_layout()
    plt.show()