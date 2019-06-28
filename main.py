import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.cluster import adjusted_rand_score

from new_fcm_pso import PSO

if __name__ == "__main__":

    glass = pd.read_csv('D:/Tsukuba/My Research/Program/MyClustering/test res pic/glass/glass.data')
    x = glass.values
    data = x[:, [3, 5]]

    #wine = datasets.load_wine()
    #x = wine.data
    #data = x[:, [0, 11]]

    #iris = datasets.load_iris()
    #original_x = iris.data
    #data = original_x[:, :2] + original_x[:, 2:]

    #data = pd.read_csv('D:/Tsukuba/My Research/Program/dataset/4_three_ciecles_with_diffR/three_ciecles_with_diffR.csv')
    #x = data.values[1:, :2]
    #y = data.values[1:, -1]
    #print(x, y)
    pso = PSO(n_cluster=6, n_particle=10, data=data)#max_iter, print_debug
    pso.run()
    ari = adjusted_rand_score(x[:, -1], pso.cluster)
    print('ARI:', ari)
    pso.show_cluter()


