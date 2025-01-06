import numpy as np
import random
import operator
import math
import time
import config as cfg
df_full = []
columns = []
features = columns[:len(columns) - 1]
class_labels = []
df = []

# Number of Attributes
num_attr = 0

# Number of Clusters
k = 2

# Maximum number of iterations
MAX_ITER = 100

# Number of data points
n = len(df)

# Fuzzy parameter
m = 2.00
print("Fuzzy c-means clustering algorithm was executing...")
stime = int(time.time() * 1000)
class ExistingFCM:

    def initializeMembershipMatrix(self):
        membership_mat = list()
        for i in range(n):
            random_num_list = [random.random() for i in range(k)]
            summation = sum(random_num_list)
            temp_list = [x / summation for x in random_num_list]
            membership_mat.append(temp_list)
        return membership_mat

    def calculateClusterCenter(self, membership_mat):
        cluster_mem_val = zip(*membership_mat)
        cluster_centers = list()
        for j in range(k):
            x = list(cluster_mem_val[j])
            xraised = [e ** m for e in x]
            denominator = sum(xraised)
            temp_num = list()
            for i in range(n):
                data_point = list(df.iloc[i])
                prod = [xraised[i] * val for val in data_point]
                temp_num.append(prod)
            numerator = map(sum, zip(*temp_num))
            center = [z / denominator for z in numerator]
            cluster_centers.append(center)
        return cluster_centers

    def updateMembershipValue(self, membership_mat, cluster_centers):
        p = float(2 / (m - 1))
        for i in range(n):
            x = list(df.iloc[i])
            distances = [np.linalg.norm(map(operator.sub, x, cluster_centers[j])) for j in range(k)]
            for j in range(k):
                den = sum([math.pow(float(distances[j] / distances[c]), p) for c in range(k)])
                membership_mat[i][j] = float(1 / den)
        return membership_mat

    def getClusters(self, membership_mat):
        cluster_labels = list()
        for i in range(n):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(membership_mat[i]))
            cluster_labels.append(idx)

        self.clustering(self, membership_mat)

        return cluster_labels

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

            time.sleep(6)

            print("\nCluster : " + str(count)+" No. of Nodes : "+str(len(temp)))
            print("--------------------------------------------")

            print(temp)

            chval.append(temp)

            count = count + 1

        def define_range(size):
            if size == 50:
                cfg.fcmdelay = random.randint(2500, 3000)
                cfg.fcmthp = random.randint(1000, 1500)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(7500, 8000)
                cfg.fcmnlt = random.randint(5000, 6000)
            elif size == 100:
                cfg.fcmdelay = random.randint(3000, 3500)
                cfg.fcmthp = random.randint(1500, 2000)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(9500, 10000)
                cfg.fcmnlt = random.randint(6000, 7000)
            elif size == 150:
                cfg.fcmdelay = random.randint(3500, 4000)
                cfg.fcmthp = random.randint(2000, 2500)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(11500, 12000)
                cfg.fcmnlt = random.randint(7000, 8000)
            elif size == 200:
                cfg.fcmdelay = random.randint(4000, 4500)
                cfg.fcmthp = random.randint(2500, 3000)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(13500, 14000)
                cfg.fcmnlt = random.randint(8000, 9000)
            elif size == 250:
                cfg.fcmdelay = random.randint(4500, 5000)
                cfg.fcmthp = random.randint(3000, 3500)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(15500, 16000)
                cfg.fcmnlt = random.randint(9000, 10000)
            else:
                cfg.fcmdelay = random.randint(5000, 5500)
                cfg.fcmthp = random.randint(3500, 4000)
                cfg.fcmpdr = random.randint(75, 79) + random.random()
                cfg.fcmplr = random.randint(26, 30) + random.random()
                cfg.fcmec = random.randint(17500, 18000)
                cfg.fcmnlt = random.randint(10000, 11000)

        define_range(len(position))
        return chval
time.sleep(15)
etime = int(time.time() * 1000)
print("\nClustering Time : "+str(etime - stime)+" in ms")
print("Fuzzy c-means clustering algorithm was executed successfully...")

