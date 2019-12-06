'''Particle component for PSO'''


import numpy as np

from fuzzyc import FCM, _obj_func


def fcm_obj_func(dataSet, U, centroids, c, m):
    n = dataSet.shape[0]
    J = 0.0
    distance = np.zeros((n, c))
    # print((dataSet.shape, centroids.shape))
    for k in range(n):
        for i in range(c):
            distance[k, i] = np.linalg.norm(dataSet[k] - centroids[i])
            J += (U[k, i] ** m) * (distance[k, i] ** 2)
    return J

class Single_Particle:

    def __init__(self,
                 n_cluster: int,
                 data,
                 w: float = 0.4,
                 c1: float = 2.0,
                 c2: float = 2.0):
        global fcm
        fcm = FCM(n_cluster=n_cluster, m=2)
        fcm.fit(data)

        self.membership = fcm._init_membership(data)  # fcm.U.copy()
        print('initial membership', self.membership)

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
        self.centroids = fcm._cal_center(data, self.membership) # fcm.center.copy()
        print('initial center', self.centroids)
        self.pbest_position = self.membership.copy()
        # print('initial pbest position:', self.pbest_position)
        self.pworst_position = self.membership.copy()
        self.best_fitness = fcm_obj_func(data, self.membership, self.centroids, self.n_cluster, 2)
        self.worst_fitness = fcm_obj_func(data, self.membership, self.centroids, self.n_cluster, 2)
        print('initial pbest fitness = J:', self.best_fitness)

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
         # print('velocity:', self.velocity)

# update the solution position and normalize membership
    def _update_membership(self, data):
        n = data.shape[0]
        new_membership = self.membership + self.velocity
        # print('new U:', new_membership)

        # when u_ki<0, make it = 0
        for k in range(n):
            #mini = np.min(new_membership[k])
            for i in range(self.n_cluster):
                #new_membership[k, i] -= mini
                if new_membership[k, i] <= 0:
                    new_membership[k, i] = 0
        #print('positive U:', new_membership)

        # normalize membership
        for k in range(n):
            summation = np.sum(new_membership[k, :])
            #print('summation:', summation)
            self.membership[k] = new_membership[k] / summation
        # print('normal U:', self.membership)

        #for _ in range(10):
        self.centroids = fcm._cal_center(data, self.membership)
        self.membership = fcm._update_membership(data, self.centroids)
        new_fitness = fcm_obj_func(data, self.membership, self.centroids, self.n_cluster, 2)

        if new_fitness < self.best_fitness:
            self.best_fitness = new_fitness
            self.pbest_position = self.membership.copy()
        if new_fitness > self.worst_fitness:
            self.worst_fitness = new_fitness
            self.pworst_position = self.membership.copy()

        # print('pbest fitness', self.best_fitness)

        # print('updated pworst:', self.pworst_position)
        # print('updated pbest:', self.pbest_position)
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
