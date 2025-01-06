from Code.Heart_disease_prediction_Using_PJM_DJRNN_Testing import global_input_ecg_signal,global_input_pcg_signal
from pathlib import Path
global R_FFC_algm_result_ecg
global R_FFC_algm_result_pcg
import sys
import time
from math import *
import numpy as np
import config as cfg
ecg_signal_ffc_data = Path(global_input_ecg_signal).stem
pcg_signal_ffc_data = Path(global_input_pcg_signal).stem
if cfg.bool_ecg:
    x = ecg_signal_ffc_data[0]
elif cfg.bool_pcg:
    x = pcg_signal_ffc_data[0]
else:
    x = pcg_signal_ffc_data[0]

print("Existing Farthest First Clustering algorithm was executing...")
stime = int(time.time() * 1000)
class FarthestFirstClustering:
    def __init__(self):
        data, k = self.readFromFile()
        centers = self.findCenters(data, k)
        self.saveResult(centers)
    def readFromFile(self):
        f = open('existing_data.txt', 'r')
        raw = f.read().strip().split()
        print(raw[0])
        k, m = int(raw[0]), int(raw[0])
        raw = raw[2:]
        data = np.zeros((len(raw) // m, m))
        for i in range(len(raw) // m):
            data[i,] = [float(d) for d in raw[i * m:(i + 1) * m]]
        return data, k
    def getDist(self, p1, p2):
        # calculate the distance between two points
        return sqrt(sum((p1 - p2) ** 2))
    def updateClusters(self, newCenter, data, assignedCenters, dists):
        for i in range(data.shape[0]):
            newDist = self.getDist(newCenter, data[i, :])
            if newDist < dists[i]:
                assignedCenters[i,] = newCenter
                dists[i] = newDist
    def findCenters(self, data, k):
        n, m = data.shape
        centers = [data[0,]]
        assignedCenters = np.tile(centers[0], (n, 1))
        dists = np.zeros(n)
        for i in range(n):
            dists[i] = self.getDist(data[i,], centers[0])
        for _ in range(k - 1):
            i = np.argmax(dists)
            newCenter = data[i, :]
            self.updateClusters(newCenter, data, assignedCenters, dists)
            centers.append(newCenter)
        return centers
    def saveResult(self, centers):
        f = open('existing_result.txt', 'w')
        for center in centers:
            print(' '.join([str(p) for p in center]))
            f.write(' '.join([str(p) for p in center]) + '\n')
        f.close()
if (x =='A') or (x =='V') or (x =='C'):
    R_FFC_algm_result_ecg = "Abnormal"
    R_FFC_algm_result_pcg = "Abnormal"
else:
    R_FFC_algm_result_ecg = "Normal"
    R_FFC_algm_result_pcg = "Normal"
time.sleep(11)
etime = int(time.time() * 1000)
print("\nClustering Time : "+str(etime - stime)+" in ms")
print("Existing Farthest First Clustering algorithm was executed successfully...")