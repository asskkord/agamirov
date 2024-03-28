import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import Birch

 # X - функция образца, Y - категория кластера образца, всего 1000 выборок, каждая выборка имеет 2 функции, всего 4 кластера, центр кластера - [-1, -1], [0,0], [1,1 ], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
                  random_state =9)

 ## Установить функцию березы
birch = Birch(n_clusters = None)
 ## Данные обучения
y_pred = birch.fit_predict(X)
 ##Рисование
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
