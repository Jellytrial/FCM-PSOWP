'''FPSO'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.seterr(divide='ignore', invalid='ignore')

from single_particle import Single_Particle, fcm_obj_func
from fuzzyc_l1 import FCM_L1

class PSO:
    def __init__(self,
                 n_cluster: int,
                 n_particle: int,
                 data,
                 hybrid: bool = True,
                 max_iter: int = 15,
                 print_debug: int = 1):
        global fcm
        fcm = FCM_L1(n_cluster=n_cluster, m=2)

        self.n_cluster = n_cluster
        self.n_particle = n_particle
        self.data = data
        self.max_iter = max_iter
        self.particles = []
        self.hybrid = hybrid
        self.print_debug = print_debug

        self.gbest_fitness = np.inf
        self.gworst_fitness = 0.0

        self.gbest_position = None
        self.gworst_position = None

        self.final_centroid = None
        self.cluster = None
        # self.gbest_sse = np.inf
        self._init_particle()

    def _init_particle(self):
        for i in range(self.n_particle):
            particle = None
            gfitness = []

            particle = Single_Particle(self.n_cluster, self.data)

            if particle.best_fitness < self.gbest_fitness:
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.best_fitness
            if particle.worst_fitness > self.gworst_fitness:
                self.gworst_position = particle.pworst_position.copy()
                self.gworst_fitness = particle.worst_fitness

            self.particles.append(particle)

            # print('Initial global best position:', self.gbest_position)
            # print('Initial global worst position:', self.gworst_position)


    def run(self):
        print('Initial global best fitness:', self.gbest_fitness)
        print('Initial global worst fitness:', self.gworst_fitness)
        history = []

        for i in range(self.max_iter):
            # FPSO
            for particle in self.particles:

                # particle.update(self.gbest_position, self.gworst_position, self.data)

                # print('in pso run pbest:', particle.pbest_position)
                # print('in pso run p fitness', particle.best_fitness)
                # set gbest and gworst
                if particle.best_fitness < self.gbest_fitness:
                    self.gbest_position = particle.pbest_position.copy()  # gbest
                    self.gbest_fitness = particle.best_fitness
                if particle.worst_fitness > self.gworst_fitness:
                    self.gworst_position = particle.pworst_position.copy()  # gworst
                    self.gworst_fitness = particle.worst_fitness

                particle.update(self.gbest_position, self.gworst_position, self.data)

                # print(i, particle.best_score, self.gbest_score)
                self.final_centroid = fcm._cal_center(self.data, self.gbest_position)
                self.gbest_position = fcm._update_membership(self.data, self.final_centroid)

            history.append(self.gbest_fitness)

            if i % self.print_debug == 0:
                print('In Iteration {:04d}/{:04d} current gbest fitness {:.13f} gworst fitness {:.13f}'.format(

                        i + 1, self.max_iter, self.gbest_fitness, self.gworst_fitness))

        #byfcm_membership = self.gbest_position.copy()
        #print('byfcm_membership:', byfcm_membership)
        #for _ in range(10):
        #    byfcm_centroid = fcm._cal_center(self.data, byfcm_membership)
        #    byfcm_membership = fcm._update_membership(self.data, byfcm_centroid)

        #self.gbest_position = byfcm_membership.copy()
        #self.final_centroid = byfcm_centroid.copy()
        #self.gbest_fitness = fcm_obj_func(self.data, self.gbest_position, self.final_centroid, self.n_cluster, 2)

        self.final_centroid = fcm._cal_center(self.data, self.gbest_position)
        self.cluster = np.argmax(self.gbest_position, axis=1)

        print('Finish with gbest score {:.18f}'.format(self.gbest_fitness))
        print('Final membership:', self.gbest_position)
        print('Final centroid', self.final_centroid)
        print('cluster:', self.cluster)
        # print(self.cluster.shape)

        # show fitness convergence
        plt.plot(history)
        plt.plot(self.max_iter)
        plt.title('convergence curve')
        plt.ylabel('fitness')
        plt.ylim(700,1400 )
        plt.xlabel('iteration')
        plt.show()

        return history



    def show_cluter(self):

        mark = ['o', 'o', 'o', 'o', '^', '+', 's', 'd', '<', 'p']
        color = ['r', 'b', 'g', 'm', 'c', 'y']
        n, m = self.data.shape

        for i in range(n):
            markIndex = int(self.cluster[i])
            plt.scatter(self.data[i, 0], self.data[i, 1], c=color[markIndex], marker=mark[markIndex], alpha = 0.6)
        for cent0, cent1 in self.final_centroid:
            plt.scatter(cent0, cent1, c='k', marker='D')
        plt.show()


if __name__ == "__mian__":
    pass

