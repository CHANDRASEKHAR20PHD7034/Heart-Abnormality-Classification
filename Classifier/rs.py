import seaborn as sns
#Making confusion matrix
from matplotlib import pyplot as plt

# ppjmdjrnncm = [[12,5],[3,11]]
# ax = sns.heatmap(ppjmdjrnncm,cbar=False, annot=True, cmap='Spectral_r', fmt='',annot_kws={"size": 12, "family":"Times New Roman"})
# ax.set_title(' Confusion Matrix for Proposed PJM-DJRNN ',fontsize=12, fontname = "Times New Roman", fontweight="bold")
# ax.set_xlabel('Predicted Values',fontsize=12, fontname = "Times New Roman", fontweight="bold")
# ax.set_ylabel('Actual Values ',fontsize=12, fontname = "Times New Roman", fontweight="bold")
# ## Ticket labels - List must be in alphabetical order
# ax.xaxis.set_ticklabels(['False', 'True'],fontsize=12, fontname = "Times New Roman")
# ax.yaxis.set_ticklabels(['False', 'True'],fontsize=12, fontname = "Times New Roman")
# plt.show()

ppjmdjrnncm = [[15, 1, 0,0], [0, 16, 0, 0], [0, 0, 15, 1], [1,0,0,15]]
ax = sns.heatmap(ppjmdjrnncm, cbar=False, annot=True, fmt='',
                                 annot_kws={"size": 12, "family": "Times New Roman"})
ax.set_title(' Confusion Matrix', fontsize=12, fontname="Times New Roman",
                             fontweight="bold")
ax.set_xlabel('Predicted', fontsize=12, fontname="Times New Roman", fontweight="bold")
ax.set_ylabel('Actual ', fontsize=12, fontname="Times New Roman", fontweight="bold")
## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['AF', 'CH','N', 'VTAB'], fontsize=12, fontname="Times New Roman")
ax.yaxis.set_ticklabels(['AF', 'CH', 'N','VTAB'], fontsize=12, fontname="Times New Roman")
plt.savefig("..//Run//Result//ConfusionMatrixProposed")
plt.show()