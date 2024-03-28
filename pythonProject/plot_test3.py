from itertools import cycle

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

## Собственный модуль итератора Python

## Создать случайный центр обработки данных
centers = [[1, 1], [-1, -1], [1, -1]]
## Количество сгенерированных данных
n_samples = 3000
## Производственные данные
X, lables_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=0.6,
                            random_state=0)

## Установить функцию иерархической кластеризации
linkages = ['ward', 'average', 'complete']
n_clusters_ = 3
ac = AgglomerativeClustering(linkage=linkages[2], n_clusters=n_clusters_)
## Данные обучения
ac.fit(X)

## Классификация каждой информации
lables = ac.labels_

##Рисование
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    ## В зависимости от того, равно ли значение в метках k, перекомпоновать массив True и False
    my_members = lables == k
    ## X [my_members, 0] Вынуть абсциссу значения, соответствующего истинному положению my_members
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
