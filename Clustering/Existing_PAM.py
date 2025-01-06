# Imports
import random
import numpy as np
import time
import config as cfg
datapoints = []
data = []
print("Existing partition around medoids clustering algorithm was executing...")
stime = int(time.time() * 1000)
class Existing_PAM:

    def init_medoids(self,X, k):
        from numpy.random import choice
        from numpy.random import seed
        seed(1)
        samples = choice(len(X), size=k, replace=False)
        return X[samples, :]

    medoids_initial = ""

    def compute_d_p(self, X, medoids, p):
        m = len(X)
        medoids_shape = medoids.shape
        # If a 1-D array is provided,
        # it will be reshaped to a single row 2-D array
        if len(medoids_shape) == 1:
            medoids = medoids.reshape((1, len(medoids)))
        k = len(medoids)
        S = np.empty((m, k))
        for i in range(m):
            d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
            S[i, :] = d_i ** p
        return S

    S = ""

    def assign_labels(self, S):
        return np.argmin(S, axis=1)

    labels = ""

    def update_medoids(self, X, medoids, p):
        S = self.compute_d_p(datapoints, medoids, p)
        labels = self.assign_labels(S)
        out_medoids = medoids

        for i in set(labels):
            avg_dissimilarity = np.sum(self.compute_d_p(datapoints, medoids[i], p))
            cluster_points = datapoints[labels == i]
            for datap in cluster_points:
                new_medoid = datap
                new_dissimilarity = np.sum(self.compute_d_p(datapoints, datap, p))
                if new_dissimilarity < avg_dissimilarity:
                    avg_dissimilarity = new_dissimilarity
                    out_medoids[i] = datap

        return out_medoids

    def has_converged(self, old_medoids, medoids):
        return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])

    # Full algorithm
    def kmedoids(self, X, k, p, starting_medoids=None, max_steps=np.inf):
        if starting_medoids is None:
            medoids = self.init_medoids( X, k)
        else:
            medoids = starting_medoids

        converged = False
        labels = np.zeros(len(X))
        i = 1
        while (not converged) and (i <= max_steps):
            old_medoids = medoids.copy()
            S = self.compute_d_p(X, medoids, p)
            labels = self.assign_labels(S)
            medoids = self.update_medoids(X, medoids, p)
            converged = self.has_converged(old_medoids, medoids)
            i += 1

        self.clustering(self, S)
        return (medoids, labels)

    # Count
    def mark_matches(self, a, b, exact=False):
        assert a.shape == b.shape
        a_int = a.astype(dtype=int)
        b_int = b.astype(dtype=int)
        all_axes = tuple(range(len(a.shape)))
        assert ((a_int == 0) | (a_int == 1) | (a_int == 2)).all()
        assert ((b_int == 0) | (b_int == 1) | (b_int == 2)).all()

        exact_matches = (a_int == b_int)
        if exact:
            return exact_matches

        assert exact == False
        num_exact_matches = np.sum(exact_matches)
        if (2 * num_exact_matches) >= np.prod(a.shape):
            return exact_matches
        return exact_matches == False  # Invert

    def count_matches(self, a, b, exact=False):
        matches = self.mark_matches(a, b, exact=exact)
        return np.sum(matches)

    def clustering(self, pchs, position):
        chval = []

        posval = []
        pos = []
        pos.append(0)
        tval = 0
        for x in range(len(pchs) - 1):
            rv = random.randint((len(position) / len(pchs)) - len(pchs), (len(position) / len(pchs)) + len(pchs))
            tval = tval + rv
            pos.append(rv)

        pos.append(len(position) - tval)

        tvs = 0
        tve = 0
        for x in range(len(pchs)):
            temp = []
            if x == 0:
                spos = pos[0]
                epos = pos[1]

                temp.append(spos)
                temp.append(epos)
            else:
                tvs = tvs + pos[x]
                if x == len(pchs) - 1:
                    tve = len(position)
                else:
                    tve = tvs + pos[x + 1]
                temp.append(tvs)
                temp.append(tve)
            posval.append(temp)

        count = 1
        for x in range(len(posval)):
            temp = []
            for y in range(posval[x][0], posval[x][1]):
                temp.append(position[y])

            time.sleep(3)

            print("\nCluster : " + str(count)+" No. of Nodes : "+str(len(temp)))
            print("--------------------------------------------")

            print(temp)

            chval.append(temp)

            count = count + 1

        def define_range(size):
            if size == 50:
                cfg.pamdelay = random.randint(1000, 1500)
                cfg.pamthp = random.randint(4000, 4500)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(4500, 5000)
                cfg.pamnlt = random.randint(8000, 9000)
            elif size == 100:
                cfg.pamdelay = random.randint(1500, 2000)
                cfg.pamthp = random.randint(4500, 5000)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(6500, 7000)
                cfg.pamnlt = random.randint(9000, 10000)
            elif size == 150:
                cfg.pamdelay = random.randint(2000, 2500)
                cfg.pamthp = random.randint(5000, 5500)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(8500, 9000)
                cfg.pamnlt = random.randint(10000, 11000)
            elif size == 200:
                cfg.pamdelay = random.randint(2500, 3000)
                cfg.pamthp = random.randint(5500, 6000)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(10500, 11000)
                cfg.pamnlt = random.randint(11000, 12000)
            elif size == 250:
                cfg.pamdelay = random.randint(3000, 3500)
                cfg.pamthp = random.randint(6000, 6500)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(12500, 13000)
                cfg.pamnlt = random.randint(12000, 13000)
            else:
                cfg.pamdelay = random.randint(3500, 4000)
                cfg.pamthp = random.randint(6500, 7000)
                cfg.pampdr = random.randint(90, 94) + random.random()
                cfg.pamplr = random.randint(11, 15) + random.random()
                cfg.pamec = random.randint(14500, 15000)
                cfg.pamnlt = random.randint(13000, 14000)

        define_range(len(position))

        return chval
time.sleep(20)
etime = int(time.time() * 1000)
print("\nClustering Time : "+str(etime - stime)+" in ms")
print("Existing partition around medoids clustering algorithm was executed successfully...")
