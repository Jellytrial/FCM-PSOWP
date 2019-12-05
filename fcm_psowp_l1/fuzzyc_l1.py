import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score

#define Manhattan distance(L1 norm)
def ManhDistance(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))

def _obj_func(data, center, u, c, m):
    n = data.shape[0]
    J = 0.0
    distance = np.zeros((n, c))
    for k in range(n):
        for i in range(c):
            distance[k, i] = np.sum(np.abs(data[k] - center[i]))
            J += (u[k, i] ** m) * distance[k, i]
    return J


class FCM_L1:

    def __init__(self,
                 n_cluster: int,
                 m,
                 max_iter: int = 15,
                 sigma: float = 1e-3):

        self.n_cluster = n_cluster
        self.m = m
        self.max_iter = max_iter
        self.sigma = sigma

        self.center = None
        self.U = None
        self.obj_func = None
        self.cluster = None

    def _init_membership(self, data):
        n = data.shape[0]
        u = np.zeros((n, self.n_cluster))
        for k in range(n):
            random_list = np.random.rand(self.n_cluster)
            summation = np.sum(random_list)
            u[k] = random_list / summation
            self.U = u

        # print('initial u in fcm:', u)
        return u

    def _cal_center(self, data, membership):
        n, p = data.shape
        center = np.zeros((self.n_cluster, p))
        new_data = np.zeros((n, p))
        u = np.empty_like(membership)

        for i in range(self.n_cluster):
            for k in range(n):
                u[k, i] = membership[k, i] ** self.m

        for j in range(p):
            new_data[:, j] = sorted(data[:, j])
            sort_order = np.argsort(data.T).T
            u_sorted = u[sort_order]
        u_sorted = u_sorted.reshape((n, self.n_cluster * p))
        # print("sorted U:", u_sorted)

        S = np.zeros((self.n_cluster * p, 1))
        for q in range(self.n_cluster * p):
            S[q] = -1 * np.sum((u_sorted[:, q] ** self.m))
        # print('pre S:', S)

        r = np.mat(np.zeros((self.n_cluster*p, 1), dtype=int))  # count
        for q in range(self.n_cluster*p):
            while (S[q] < 0):
                S[q, 0] = S[q, 0] + 2 * (u_sorted[r[q], q] ** self.m)
                r[q] = r[q] + 1
        r = r.reshape((p, self.n_cluster))
        # print('new S:', S)
        # print('r', r)

        for i in range(self.n_cluster):
            for j in range(p):
                center[i, j] = new_data[r[j, i], j]
        # print('cluster center', center)

        return center


    def _update_membership(self, data, center):
        n = data.shape[0]
        t = - (1 / (self.m - 1))
        u = np.zeros((n, self.n_cluster))
        distance = np.zeros((n, self.n_cluster))
        for k in range(n):
            for i in range(self.n_cluster):
                distance[k][i] = np.sum(np.abs(data[k] - center[i]))
        # print('d:', distance)

        for k in range(n):
            for i in range(self.n_cluster):
                u[k, i] = (distance[k, i] ** t) / np.sum(distance[k, :] ** t)
        # print('new u', u)
        return u


    def _assign_cluster(self):
        self.cluster = np.argmax(self.U, axis=1)
        return self.cluster

    def fit(self, data):
        self.U = self._init_membership(data)
        count = 0
        while count <= self.max_iter:
            self.center = self._cal_center(data, self.U)
            #print('process center:', self.center)
            self.U = self._update_membership(data, self.center)
            #print('u:', self.U)
            count += 1
        self.obj_func = _obj_func(data, self.center, self.U, self.n_cluster, self.m)


        # print('center:', self.center)
        # print('membership:', self.U)
        # print('J:', self.obj_func)
        return self

    def _predict(self):
        return self._assign_cluster(self.U)


if __name__ == "__main__":
    #pass
    glass = pd.read_csv('D:/Tsukuba/My Research/Program/MyClustering/test res pic/glass/glass.data')
    x = glass.values
    data = x[:, [3, 5]]

    #wine = datasets.load_wine()
    #x = wine.data
    #data = x[:, [0, 11]]
    fcm = FCM(n_cluster=3, m=2)
    fcm.fit(data)
    cluster = fcm._assign_cluster()
    ari = adjusted_rand_score(x[:, -1], cluster)
    print('ARI:', ari)

    #dataset = pd.read_csv('D:/Tsukuba/My Research/Program/dataset/1_normal_data/normal_data.csv')
    #data = dataset.values
    #print(data[: 5])

    #fcm = FCM(n_cluster=4, m=2)
    #fcm.fit(data)
