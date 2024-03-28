import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

 # X - функция образца, Y - категория кластера выборки, всего 1000 выборок, каждая выборка имеет 2 характеристики, всего 4 кластера, центр кластера находится в [-1, -1], [0,0], [1,1 ], [2,2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.3, 0.4, 0.3],
                  random_state = 0)

 ## Установить функцию gmm
gmm = GaussianMixture(n_components=4, covariance_type='full').fit(X)
 ## Данные обучения
y_pred = gmm.predict(X)

 ##Рисование
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()
