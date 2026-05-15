# Гайд по лабе 5 — Кластеризация

Здесь объяснено всё что нужно для лабы: что такое каждый алгоритм, как работает, какие параметры крутить, что писать в выводе.

---

## Что такое кластеризация

Обучение **без учителя** — у тебя нет меток классов. Алгоритм сам ищет группы (кластеры) похожих объектов.

В отличие от классификации:
- Классификация: есть `y`, учим модель предсказывать метки
- Кластеризация: нет `y`, ищем структуру в данных

---

## Блок 1 — Генерация данных

### make_blobs — для кластеризации

Генерирует точки вокруг центроид. Идеален для кластеризации — кластеры чёткие.

```python
from sklearn.datasets import make_blobs

X, y_true = make_blobs(
    n_samples=300,     # кол-во точек
    n_features=2,      # размерность (2 для визуализации)
    centers=4,         # кол-во кластеров
    cluster_std=0.8,   # разброс внутри кластера (меньше = чётче)
    random_state=42
)
```

`y_true` — истинные метки кластеров, нужны потом для **внешних метрик**.

Сделай 2 датасета с разным `centers` (например 3 и 5) и разным `cluster_std`.

### make_classification — для классификации

Генерирует данные для задачи классификации. Кластеры менее чёткие.

```python
from sklearn.datasets import make_classification

X, y_true = make_classification(
    n_samples=300,
    n_features=4,
    n_classes=3,           # кол-во классов
    n_clusters_per_class=1,
    random_state=42
)
```

Сделай 3 датасета с разными `n_features`, `n_classes`.

### Свой датасет (stud_end.csv)

```python
df = pd.read_csv('stud_end.csv')
y_true = df['Target']           # сохраняем для внешних метрик
X = df.drop(columns=['Target']) # удаляем метку
```

Не забудь **масштабирование** — большинство алгоритмов чувствительны к масштабу:

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)
```

---

## Блок 2 — KMeans

### Как работает

1. Случайно ставит K центроид
2. Каждую точку относит к ближайшей центроиде
3. Пересчитывает центроиды как среднее кластера
4. Повторяет шаги 2-3 до сходимости

**WCSS** (Within-Cluster Sum of Squares) — сумма квадратов расстояний от точек до своих центроид. Чем меньше — тем лучше, но при K=N будет 0 (бессмысленно).

### Код

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_sc)

kmeans.inertia_         # WCSS
kmeans.cluster_centers_ # координаты центроид
```

### Подбор K — метод локтя

```python
wcss = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_sc)
    wcss.append(km.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel('K')
plt.ylabel('WCSS')
plt.title('Метод локтя')
plt.show()
```

Ищи "локоть" — точку где WCSS резко замедляет падение.

### Подбор K — силуэт

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)
    scores.append(silhouette_score(X_sc, labels))

plt.plot(range(2, 11), scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette')
plt.title('Метод силуэта')
plt.show()
```

Ищи максимум. Финальный K выбираешь исходя из обоих графиков.

---

## Блок 3 — Иерархическая кластеризация

### Как работает

**Агломеративный** подход (снизу вверх):
1. Каждая точка — отдельный кластер
2. Объединяем два ближайших кластера
3. Повторяем до одного кластера

Результат — **дендрограмма** (дерево), по которой видно оптимальное K.

```python
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# Дендрограмма
plt.figure(figsize=(14, 6))
dend = sch.dendrogram(sch.linkage(X_sc, method='ward'))
plt.title('Дендрограмма')
plt.show()
# Смотри где самые длинные вертикальные линии — там резать (= оптимальный K)

# Сама кластеризация
hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
labels = hc.fit_predict(X_sc)
```

**linkage** — метод расстояния между кластерами:
- `ward` — минимизирует WCSS (лучший выбор по умолчанию)
- `complete` — максимальное расстояние
- `average` — среднее расстояние

---

## Блок 4 — DBSCAN

### Как работает

Не требует задавать K. Ищет кластеры как плотные области точек.

Параметры:
- `eps` — радиус окрестности точки
- `min_samples` — минимум соседей чтобы быть "ядром"

Точки которые не попали ни в один кластер → **шум** (метка `-1`).

### Когда использовать

- Кластеры произвольной формы (не обязательно круглые)
- Есть выбросы
- Не знаешь K

### Код

```python
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.5, min_samples=5)
labels = db.fit_predict(X_sc)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f'Кластеров: {n_clusters}, шумовых точек: {n_noise}')
```

### Подбор eps

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_sc)
distances, _ = nn.kneighbors(X_sc)
distances = np.sort(distances[:, -1])

plt.plot(distances)
plt.ylabel('5-е расстояние до соседа')
plt.title('k-distance graph для подбора eps')
plt.show()
# Ищи "локоть" — это и есть хороший eps
```

**Важно:** DBSCAN часто даёт плохой силуэт и ARI на stud_end — это нормально, просто честно напиши в выводе.

---

## Блок 5 — EM-алгоритм (GaussianMixture)

### Как работает

Предполагает что данные — смесь K гауссовых распределений. Находит параметры каждого (среднее, ковариация).

В отличие от KMeans:
- KMeans: жёсткое присвоение (точка принадлежит ровно одному кластеру)
- EM: мягкое присвоение (вероятность принадлежности к каждому кластеру)

**E-шаг** — вычислить вероятности принадлежности  
**M-шаг** — обновить параметры распределений  
Повторять до сходимости.

### Код

```python
from sklearn.mixture import GaussianMixture

gm = GaussianMixture(n_components=4, random_state=42)
labels = gm.fit_predict(X_sc)

gm.means_          # центры кластеров
gm.covariances_    # ковариационные матрицы
```

Подбор количества компонент — через **BIC** (меньше = лучше):

```python
bic = []
for k in range(2, 11):
    gm = GaussianMixture(n_components=k, random_state=42)
    gm.fit(X_sc)
    bic.append(gm.bic(X_sc))

plt.plot(range(2, 11), bic, marker='o')
plt.xlabel('K')
plt.ylabel('BIC')
plt.title('Подбор K для GaussianMixture')
plt.show()
```

---

## Блок 6 — Affinity Propagation

### Как работает

Не требует задавать K. Точки "голосуют" за то, кто из них станет "представителем" кластера. Количество кластеров определяется автоматически.

Параметры:
- `damping` — затухание (0.5–0.99), влияет на сходимость
- `preference` — чем меньше, тем меньше кластеров

### Код

```python
from sklearn.cluster import AffinityPropagation

ap = AffinityPropagation(damping=0.7, random_state=42)
labels = ap.fit_predict(X_sc)

n_clusters = len(ap.cluster_centers_indices_)
print(f'Найдено кластеров: {n_clusters}')
```

**Предупреждение:** на больших датасетах работает медленно и может найти слишком много кластеров. На stud_end может выдать 30+ кластеров — это ок, просто опиши в выводе.

---

## Блок 7 — Визуализация

Если признаков больше 2 — используй PCA для визуализации:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_sc)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, label='Кластер')
plt.title('KMeans — визуализация через PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()
```

Если 2 признака — рисуй напрямую без PCA.

---

## Блок 8 — CustomKMeans

Напиши свой класс. Минимальная реализация:

```python
import numpy as np

class CustomKMeans:
    def __init__(self, n_clusters=3, max_iter=300, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        rng = np.random.default_rng(self.random_state)
        # Случайно инициализируем центроиды
        idx = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx].copy()

        for _ in range(self.max_iter):
            # E-шаг: присваиваем метки
            labels = self._assign(X)
            # M-шаг: пересчитываем центроиды
            new_centers = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Проверяем сходимость
            if np.allclose(self.cluster_centers_, new_centers):
                break
            self.cluster_centers_ = new_centers

        self.labels_ = self._assign(X)
        self.inertia_ = self._wcss(X)
        return self

    def predict(self, X):
        return self._assign(X)

    def fit_predict(self, X):
        return self.fit(X).labels_

    def _assign(self, X):
        # Расстояния от каждой точки до каждой центроиды
        dists = np.linalg.norm(X[:, None] - self.cluster_centers_[None, :], axis=2)
        return np.argmin(dists, axis=1)

    def _wcss(self, X):
        labels = self._assign(X)
        return sum(
            np.sum((X[labels == k] - self.cluster_centers_[k]) ** 2)
            for k in range(self.n_clusters)
        )
```

---

## Блок 9 — Метрики

### Внутренние (не нужен y_true)

**Silhouette Score** — насколько точка близка к своему кластеру vs чужим. От -1 до 1, чем больше тем лучше.

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_sc, labels)
```

**Davies-Bouldin Index** — среднее сходство кластера с наиболее похожим на него. Чем **меньше** тем лучше.

```python
from sklearn.metrics import davies_bouldin_score
score = davies_bouldin_score(X_sc, labels)
```

### Внешние (нужен y_true)

Используй `y_true` который сохранил в начале.

**Adjusted Rand Index (ARI)** — насколько разбиение на кластеры совпадает с истинными метками. От -1 до 1, чем больше тем лучше. 0 = случайно, 1 = идеально.

```python
from sklearn.metrics import adjusted_rand_score
score = adjusted_rand_score(y_true, labels)
```

**Normalized Mutual Information (NMI)** — информация о взаимосвязи кластеров и истинных меток. От 0 до 1.

```python
from sklearn.metrics import normalized_mutual_info_score
score = normalized_mutual_info_score(y_true, labels)
```

**Важно:** для DBSCAN шумовые точки (метка -1) портят внешние метрики. Можно посчитать только по точкам у которых `labels != -1`, но в таблицу ставь полный результат.

---

## Блок 10 — Описание кластеров (stud_end)

```python
df['cluster'] = labels  # добавляем колонку

# Средние значения признаков по кластерам
cluster_profile = df.groupby('cluster').mean().round(2)
print(cluster_profile)

# Размер кластеров
print(df['cluster'].value_counts().sort_index())
```

По этому можно описать кластеры словами, например:
- Кластер 0: молодые студенты, низкий доход семьи, высокий % отчислений
- Кластер 1: ...

---

## Блок 11 — Итоговая таблица

```python
results = []
algorithms = {
    'KMeans (sklearn)': labels_kmeans,
    'Hierarchical':     labels_hc,
    'DBSCAN':           labels_dbscan,
    'GaussianMixture':  labels_gm,
    'AffinityProp':     labels_ap,
    'CustomKMeans':     labels_custom,
}

for name, labs in algorithms.items():
    # Silhouette не считается если 1 кластер или все точки — шум
    mask = labs != -1
    sil = silhouette_score(X_sc[mask], labs[mask]) if mask.sum() > 1 and len(set(labs[mask])) > 1 else None
    results.append({
        'Алгоритм': name,
        'Silhouette': round(sil, 4) if sil else '-',
        'Davies-Bouldin': round(davies_bouldin_score(X_sc[mask], labs[mask]), 4) if sil else '-',
        'ARI': round(adjusted_rand_score(y_true, labs), 4),
        'NMI': round(normalized_mutual_info_score(y_true, labs), 4),
    })

pd.DataFrame(results).set_index('Алгоритм')
```

---

## Блок 12 — Вывод

Структура вывода:

1. Какие алгоритмы использовались и на каких данных
2. Какой алгоритм лучший для синтетики и почему (с цифрами из таблицы)
3. Какой алгоритм лучший для stud_end и почему
4. Что интересного нашлось в кластерах stud_end (описание групп)
5. Ограничения: почему DBSCAN / AP плохо сработали (если так)

---

## Частые ошибки

| Проблема | Решение |
|---|---|
| `ValueError: Number of labels is 1` в силуэте | DBSCAN нашёл 1 кластер — уменьши eps |
| ARI отрицательный | Это нормально, значит хуже случайного |
| AP не сходится | Увеличь `max_iter=500`, `damping=0.9` |
| WCSS = 0 у CustomKMeans | n_clusters равен числу точек — проверь данные |
| FutureWarning у AgglomerativeClustering | Замени `affinity=` на `metric=` |
