import pandas as pd
import numpy as np
# np.set_printoptions(threshold=np.inf)
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score, calinski_harabaz_score

from new_fcm_pso import PSO


def loadData(filename):
    dataSet = np.mat(np.genfromtxt(filename, dtype='float64', delimiter=','))
    print(dataSet.shape)
    return dataSet


if __name__ == "__main__":

    #glass = pd.read_csv('D:/Tsukuba/My Research/Program/MyClustering/test res pic/glass/glass.data')
    #x = glass.values
    #data = x[:, [3, 5]]

    #wine = datasets.load_wine()
    #x = wine.data
    #data = x[:, [0, 11]]

    #iris = datasets.load_iris()
    #original_x = iris.data
    #data = original_x[:, :2] + original_x[:, 2:]

    # artificle data
    data = pd.read_csv('D:/Tsukuba/My Research/Program/dataset/diff_var.csv')
    x = data.values[:, 1:3]
    y = data.values[:, 0]

    pso = PSO(n_cluster=3, n_particle=10, data=x)  # max_iter, print_debug
    pso.run()
    pred_cluster = pso.cluster

    ari = adjusted_rand_score(y, pred_cluster)
    ch = calinski_harabaz_score(x, pred_cluster)
    print('ARI:', ari)
    print('CH:', ch)
    pso.show_cluter()


