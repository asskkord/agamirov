# -*- coding:utf-8 -*-
from itertools import cycle
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt


 ## Создать случайный центр обработки данных
centers = [[1, 1], [-1, -1], [1, -1]]
 ## Количество сгенерированных данных
n_samples=10000
 ## Производственные данные
X, _ = make_blobs(n_samples=n_samples, centers= centers, cluster_std=0.6,
                  random_state =0)

 ## Пропускная способность, то есть радиус поиска, когда определенная точка является ядром
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
 ## Установите функцию среднего сдвига
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
 ## Данные обучения
ms.fit(X)
 ## Метка для каждой точки
labels = ms.labels_
print(labels)
 ## Набор точек в центре кластера
cluster_centers = ms.cluster_centers_
print('cluster_centers:',cluster_centers)
 ## Общая классификация тегов
labels_unique = np.unique(labels)
 ## Количество кластеров, то есть количество категорий
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)


 ##Рисование
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
         ## В зависимости от того, равно ли значение в метках k, перекомпоновать массив True и False
    my_members = labels == k
    cluster_center = cluster_centers[k]
         ## X [my_members, 0] Вынуть абсциссу значения, соответствующего истинному положению my_members
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()


