'''Particle component for FPSO
1. cal cluster center
2. cal L1fcm objective fuc J
3. set pbest and pworst
4. update velocity of each particle
5. update position(membership) of each particle'''


import numpy as np

from fuzzyc_l1 import FCM_L1, _obj_func


def fcm_obj_func(dataSet, U, centroids, c, m):
    n = dataSet.shape[0]
    J = 0.0
    distance = np.zeros((n, c))
    # print((dataSet.shape, centroids.shape))
    for k in range(n):
        for i in range(c):
            distance[k, i] = np.sum(np.abs(dataSet[k] - centroids[i]))
            J += (U[k, i] ** m) * distance[k, i]
    return J

class Single_Particle:

    def __init__(self,
                 n_cluster: int,
                 data,
                 w: float = 0.4,
                 c1: float = 2.0,
                 c2: float = 2.0):
        global fcm
        fcm = FCM_L1(n_cluster=n_cluster, m=2)
        fcm.fit(data)

        # initialize constants
        self._w = w
        self._c1 = c1
        self._c2 = c2
        self.r1 = np.random.uniform()
        self.r2 = np.random.uniform()
        self.r3 = np.random.normal(0, 1)
        self.r4 = np.random.normal(0, 1)

        # print(self.r1, self.r2, self.r3, self.r4)
        self.n_cluster = n_cluster

        # initialize particle
        self.membership = fcm._init_membership(data)  # fcm.U.copy()
        print('initial membership of each particle:', self.membership)

        self.pbest_position = self.membership.copy()
        # print('initial pbest position:', self.pbest_position)
        self.pworst_position = self.membership.copy()

        self.centroids = fcm._cal_center(data, self.membership)  # fcm.center.copy()
        print('initial center of each particle:', self.centroids)

        self.best_fitness = fcm_obj_func(data, self.pbest_position, self.centroids, self.n_cluster, 2)
        self.worst_fitness = fcm_obj_func(data, self.pworst_position, self.centroids, self.n_cluster, 2)
        print('initial pbest fitness = J:', self.best_fitness)
        print('initial pworst fitness = J:', self.worst_fitness)

        self.velocity = np.zeros_like(self.membership)


    def update(self, gbest_position, gworst_position, data):
        '''Update particle' velocity and centroids
        Parameter
        ----------------
        gbest_position
        data'''

        self._update_velocity(gbest_position, gworst_position)
        self._update_membership(data)


    def _update_velocity(self, gbest_position, gworst_position):
         '''Update velocity based on previous value, 
         cognitive component, and social component'''

         v_old = self._w * self.velocity

         cognitive_component = self._c1 * self.r1 * (self.pbest_position - self.membership) +\
                               self.r3 * (self.pworst_position - self.membership)

         social_component = self._c2 * self.r2 * (gbest_position - self.membership) +\
                            self.r4 * (gworst_position - self.membership)

         self.velocity = v_old + cognitive_component + social_component

         return self
         # print('velocity:', self.velocity)

# update the solution position and normalize membership by FCM
    def _update_membership(self, data):
        n = data.shape[0]
        new_membership = self.membership + self.velocity
        # print('new U:', new_membership)

        # when u_ki<0, make it = 0
        for k in range(n):
            for i in range(self.n_cluster):
                if new_membership[k, i] <= 0:
                    new_membership[k, i] = 0

        zero_list = np.zeros((1, self.n_cluster))
        for k in range(n):
            if (new_membership[k, :] == zero_list).all():
                # print('k:', k)
                new_membership[k, :] = np.random.uniform(0, 1, (1, self.n_cluster))
            else:
                pass

        # print('positive U:', new_membership)

        # normalize membership
        new_u = np.empty_like(self.membership)
        for k in range(n):
            summation = np.sum(new_membership[k, :])
            #print('summation:', summation)
            new_u[k] = new_membership[k] / summation
        #print('normal U:', self.membership)

        new_centroids = fcm._cal_center(data, new_u)
        update_membership = fcm._update_membership(data, new_centroids)

        new_fitness = fcm_obj_func(data, update_membership, new_centroids, self.n_cluster, 2)

        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.pbest_position = update_membership.copy()
        if new_fitness > self.worst_fitness:
            self.worst_fitness = new_fitness
            self.pworst_position = update_membership.copy()

        # print('pbest fitness', self.best_fitness)
        # print('updated pbest:', self.pbest_position)

        # print('pworst fitness', self.worst_fitness)
        # print('updated pworst:', self.pworst_position)

        return self


    def _predict(self):
        '''Predict new data's cluster using minimum distance to centroid
        '''

        cluster = self._assign_cluster()
        #print('cluster', cluster)
        return cluster


    def _assign_cluster(self):
        cluster = np.argmax(self.membership, axis=1)
        return cluster


if __name__ == "__main__":

    pass
