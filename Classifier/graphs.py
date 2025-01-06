import numpy as np
from matplotlib import pyplot as plt

''' Computational time'''
# fig = plt.figure(figsize=(10, 6))
# X = ['Proposed PJM-DJRNN', 'DJRNN', 'RNN', 'DNN', 'ANN', 'ENN']
# entime = [(18003), (28001), (40004), (58015), (66016), (78008)]
# X_axis = np.arange(len(X))
# # clr = ['deeppink','cadetblue','orange','darkgreen','lightcoral']
# plt.plot(X, entime, color='deeppink', marker="*")
# plt.xticks(X_axis, X, font="Times New Roman")
# plt.yticks(font="Times New Roman")
# plt.xlabel("Techniques", font="Times New Roman", fontweight="bold")
# plt.ylabel("Computational time (ms)", font="Times New Roman", fontweight="bold")
# # plt.title("Key Generation time", font="Times New Roman", fontweight="bold")
# plt.savefig("..\\..\\Run\\Result\\Computation_time.png")
# plt.show()

''' Clustering time'''
# fig = plt.figure(figsize=(10, 6))
# X = ['Proposed RFFC', 'FFC', 'FCM', 'Kmeans', 'PAM']
# entime = [(3245), (11002), (15001), (18008), (20004)]
# X_axis = np.arange(len(X))
# # clr = ['deeppink','cadetblue','orange','darkgreen','lightcoral']
# plt.plot(X, entime, color='darkgreen', marker="*")
# plt.xticks(X_axis, X, font="Times New Roman")
# plt.yticks(font="Times New Roman")
# plt.xlabel("Techniques", font="Times New Roman", fontweight="bold")
# plt.ylabel("Clustering time (ms)", font="Times New Roman", fontweight="bold")
# # plt.title("Key Generation time", font="Times New Roman", fontweight="bold")
# plt.savefig("..\\..\\Run\\Result\\Clustering_time.png")
# plt.show()

''' Clustering accuracy'''
# fig = plt.figure(figsize=(10, 6))
# X = ['Proposed RFFC', 'FFC', 'FCM', 'Kmeans', 'PAM']
# entime = [(91.0066), (85.4481), (82.9932), (78.1002), (72.6883)]
# X_axis = np.arange(len(X))
# clr = ['deeppink','cadetblue','orange','darkgreen','lightcoral']
# plt.bar(X, entime, color=clr)
# plt.xticks(X_axis, X, font="Times New Roman")
# plt.yticks(font="Times New Roman")
# plt.xlabel("Techniques", font="Times New Roman", fontweight="bold")
# plt.ylabel("Clustering accuracy (%)", font="Times New Roman", fontweight="bold")
# # plt.title("Key Generation time", font="Times New Roman", fontweight="bold")
# plt.savefig("..\\..\\Run\\Result\\Clustering_accuracy.png")
# plt.show()

''' PSNR'''

# fig = plt.figure(figsize=(10, 6))
# X = ['Proposed BrF-BLF', 'BLF', 'BWF', 'CF', 'LBPF']
# entime = [(36.1091), (31.56254), (28.3088), (21.8116), (19.4942)]
# X_axis = np.arange(len(X))
# clr = ['chocolate','darkolivegreen','deepskyblue','orchid','darkslateblue']
# plt.bar(X, entime, color=clr)
# plt.xticks(X_axis, X, font="Times New Roman")
# plt.yticks(font="Times New Roman")
# plt.xlabel("Techniques", font="Times New Roman", fontweight="bold")
# plt.ylabel("PSNR (db)", font="Times New Roman", fontweight="bold")
# # plt.title("Key Generation time", font="Times New Roman", fontweight="bold")
# plt.savefig("..\\..\\Run\\Result\\PSNR_ECG.png")
# plt.show()

''' Fitness vs. iteration'''

N = 5
ind = np.arange(N)
width = 0.13

ProposedPDFSLO = [88, 91, 141, 191, 204]
bar1 = plt.bar(ind, ProposedPDFSLO, width)
ExistingSLO = [76, 78, 132, 189, 199]
bar2 = plt.bar(ind + width, ExistingSLO, width)
ExistingAO = [59, 71, 121, 177, 190]
bar3 = plt.bar(ind + width * 2, ExistingAO, width)
ExistingDSO = [54, 58, 91, 124, 132]
bar4 = plt.bar(ind + width * 3, ExistingDSO, width)
ExistingSSO = [35, 47, 68, 98, 121]
bar5 = plt.bar(ind + width * 4, ExistingSSO, width)
# plt.ylim(0, 170)

plt.xlabel("Iterations", fontname='Times New Roman', fontsize=12, fontweight="bold")
plt.ylabel('Fitness', fontname='Times New Roman', fontsize=12, fontweight="bold")
# plt.title("Results Comparison",fontname='Times New Roman', fontsize=12,fontweight="bold")
colors = ['cornflowerblue', 'lightpink', 'wheat', 'navajowhite', 'paleturquoise']
plt.xticks(ind + width,
                    ["10", "20", "30", "40", "50"], fontname='Times New Roman', fontsize=12)
plt.yticks(fontname='Times New Roman', fontsize=12)
plt.legend((bar1, bar2, bar3, bar4, bar5),
                                   ('Proposed PDF-SLO', 'SLO', 'AO', 'DSO', 'SSO'),
                                   prop={'family': 'Times New Roman', 'size': 12}, loc='upper left')
plt.savefig("..\\..\\Run\\Result\\fitnessVsiteration.png")
plt.show()
