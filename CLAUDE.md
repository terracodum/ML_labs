# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of ML labs and practicals implemented as Jupyter notebooks. Each folder is self-contained with its own dataset copies.

## Running notebooks

```bash
jupyter notebook          # открыть браузерный интерфейс
jupyter lab               # альтернатива
jupyter nbconvert --to script lab1/lab1_class.ipynb  # экспорт в .py
```

## Structure

| Папка | Задача | Датасет |
|-------|--------|---------|
| `lab1/` | Классификация (`lab1_class.ipynb`) и регрессия (`lab1_reg.ipynb`) | `cars_end.csv`, `stud_end.csv` |
| `lab2/` | Регрессия: LinearRegression, Lasso, Ridge, ElasticNet + hyperparameter search | `cars_end.csv` |
| `prac1/` | EDA (разведочный анализ) | `forestfires.csv` |
| `prac2/` | Парсинг данных + анализ CPI | `Dataset1.csv`, `Practice2_Harlov_CPI.csv` |
| `Examples/` | Примеры EDA | `cars.ipynb` |

Исходные `cars.csv` и `students.csv` в корне — исходники; обработанные версии (`_end.csv`) лежат в папках лабораторных.

## Common patterns

**Pipeline для регрессии** (из lab2):
```python
Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', model)
])
```

**Метрики** используются: MAE, MSE, RMSE, MAPE, R2.

**Подбор гиперпараметров**: GridSearchCV, RandomizedSearchCV, Optuna (`optuna.create_study`).

## Key libraries

`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `optuna`, `tqdm`
