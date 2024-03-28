from itertools import cycle

from sklearn.datasets import make_blobs
from sklearn.cluster import spectral_clustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

## Собственный модуль итератора Python

## Создать случайный центр обработки данных
centers = [[1, 1], [-1, -1], [1, -1]]
## Количество сгенерированных данных
n_samples = 3000
## Производственные данные
X, lables_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.6,
                            random_state=0)

## Преобразовать в матрицу, вход должен быть симметричной матрицей
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(X)).astype(np.int32)
metrics_metrix += -1 * metrics_metrix.min()
## Установите функцию спектральной кластеризации
n_clusters_ = 4
lables = spectral_clustering(metrics_metrix, n_clusters=n_clusters_)

##Рисование
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    ## В зависимости от того, равно ли значение в метках k, перекомпонуйте массив True и False
    my_members = lables == k
    ## X [my_members, 0] Вынуть абсциссу значения, соответствующего истинному положению my_members
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
