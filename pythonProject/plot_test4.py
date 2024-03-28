from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

## Собственный модуль итератора Python
from sklearn.preprocessing import StandardScaler

## Создать случайный центр обработки данных
centers = [[1, 1], [-1, -1], [1, -1]]
## Количество сгенерированных данных
n_samples = 750
## Производственные данные: на результаты этого эксперимента влияет cluster_std или разница между eps и cluster_std
X, lables_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.4,
                            random_state=0)

## Установить функцию иерархической кластеризации
db = DBSCAN(eps=0.3, min_samples=10)
## Данные обучения
db.fit(X)
## Инициализировать массив типа bool, который имеет значение False
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
'''
       Вот ключевые моменты (для этой строки кода: xy = X [class_member_mask & ~ core_samples_mask]):
       db.core_sample_indices_ представляет точку, которая временно помечается как точка шума в процессе поиска набора базовых точек (т. е. окружающие точки
       Менее min_samples), а не окончательная точка шума. В процессе соединения основных точек эти точки будут переклассифицированы (т.е.
       Это не будет означать -1) для точек шума.Также можно понять, что эти точки не подходят для точек ядра, но они будут включены в диапазон определенной точки ядра.
'''
core_samples_mask[db.core_sample_indices_] = True

## Классификация каждой информации
lables = db.labels_

## Количество категорий: метки содержат -1, что указывает на точки шума
n_clusters_ = len(np.unique(lables)) - (1 if -1 in lables else 0)

##Рисование
unique_labels = set(lables)
'''
       1) np.linspace возвращает количество len (unique_labels) между [0,1]
       2) plt.cm модуль сопоставления цветов
       3) Каждый сгенерированный цвет содержит 4 значения, которые являются rgba
       4) Фактически, эта строка кода означает создание 4 значений цвета, которые могут соответствовать спектру
'''
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

plt.figure(1)
plt.clf()

for k, col in zip(unique_labels, colors):
    ## - 1 означает точку шума, здесь k означает черный
    if k == -1:
        col = 'k'

        ## Сгенерируйте массив True и False и установите для меток == k значение True
    class_member_mask = (lables == k)

    ## Два массива выполняют & операцию, чтобы найти значение, которое является основной точкой и равно категории k markeredgecolor = 'k',
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=14)
    '''
               1) ~ Приоритет наивысший, core_samples_mask побитовое инвертирование, и положение точки шума получается
               2) & После операции найдите положение точки шума в начале, но переклассифицированная точка принадлежит k
               3) Расширение после классификации ядра
    '''
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', c=col, markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
