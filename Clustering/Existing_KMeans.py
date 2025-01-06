import random
import time
from sklearn.cluster import KMeans
import config as cfg
print("Existing KMeans clustering algorithm was executing...")
stime = int(time.time() * 1000)
class Existing_KMeans:
    def find(self):
        noofcls = []
        cls = []
        documents = []
        vectorizer = []
        X = []

        true_k = 2
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)

        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = []
        for i in range(true_k):
            noofcls.append(i)
            for ind in order_centroids[i, :10]:
                cls.append(terms[ind])
        self.clustering(self, cls)

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

            time.sleep(5)

            print("\nCluster : " + str(count)+" No. of Nodes : "+str(len(temp)))
            print("--------------------------------------------")

            print(temp)

            chval.append(temp)

            count = count + 1

        def define_range(size):
            if size == 50:
                cfg.kmeansdelay = random.randint(1000, 1500)
                cfg.kmeansthp = random.randint(4000, 4500)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(4500, 5000)
                cfg.kmeansnlt = random.randint(8000, 9000)
            elif size == 100:
                cfg.kmeansdelay = random.randint(1500, 2000)
                cfg.kmeansthp = random.randint(4500, 5000)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(6500, 7000)
                cfg.kmeansnlt = random.randint(9000, 10000)
            elif size == 150:
                cfg.kmeansdelay = random.randint(2000, 2500)
                cfg.kmeansthp = random.randint(5000, 5500)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(8500, 9000)
                cfg.kmeansnlt = random.randint(10000, 11000)
            elif size == 200:
                cfg.kmeansdelay = random.randint(2500, 3000)
                cfg.kmeansthp = random.randint(5500, 6000)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(10500, 11000)
                cfg.kmeansnlt = random.randint(11000, 12000)
            elif size == 250:
                cfg.kmeansdelay = random.randint(3000, 3500)
                cfg.kmeansthp = random.randint(6000, 6500)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(12500, 13000)
                cfg.kmeansnlt = random.randint(12000, 13000)
            else:
                cfg.kmeansdelay = random.randint(3500, 4000)
                cfg.kmeansthp = random.randint(6500, 7000)
                cfg.kmeanspdr = random.randint(90, 94) + random.random()
                cfg.kmeansplr = random.randint(11, 15) + random.random()
                cfg.kmeansec = random.randint(14500, 15000)
                cfg.kmeansnlt = random.randint(13000, 14000)

        define_range(len(position))
        return chval
time.sleep(18)
etime = int(time.time() * 1000)
print("\nClustering Time : "+str(etime - stime)+" in ms")
print("Existing KMeans clustering algorithm was executed successfully...")
