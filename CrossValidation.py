import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.30

ProposedPELSFDCNN = [100, 100, 100, 100, 100]
ExistingDCNN = [97.23,	97.89, 98.48, 98.89, 98.91]

br1 = np.arange(len(ProposedPELSFDCNN))
br2 = [x + barWidth for x in br1]

plt.bar(br1, ProposedPELSFDCNN, color ='#1E90FF', width = barWidth, edgecolor ='grey', label ='Training')
plt.bar(br2, ExistingDCNN, color ='darksalmon', width = barWidth, edgecolor ='grey', label ='Validation')
plt.xticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)
plt.yticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)

plt.title("", fontweight='bold', fontname="Times New Roman", fontsize=14)
plt.xlabel('K-Fold', fontweight='bold', fontname="Times New Roman", fontsize=14)
plt.ylabel('Accuracy (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
plt.xticks([r + barWidth for r in range(len(ProposedPELSFDCNN))], ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold'])

plt.rcParams['font.sans-serif'] = "Times New Roman"
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'
plt.legend(loc=4)


plt.savefig("ECG_CrossValidation.png")
# plt.show()
plt.close()

def PCG():
    import numpy as np
    import matplotlib.pyplot as plt

    barWidth = 0.30

    ProposedPELSFDCNN = [100, 100, 100, 100, 100]
    ExistingDCNN = [96.73, 97.89, 98.58, 98.99, 98.81]

    br1 = np.arange(len(ProposedPELSFDCNN))
    br2 = [x + barWidth for x in br1]

    plt.bar(br1, ProposedPELSFDCNN, color='#008080', width=barWidth, edgecolor='grey', label='Training')
    plt.bar(br2, ExistingDCNN, color='#EED2EE', width=barWidth, edgecolor='grey', label='Validation')
    plt.xticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)
    plt.yticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)

    plt.title("", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xlabel('K-Fold', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([r + barWidth for r in range(len(ProposedPELSFDCNN))],
               ['1st Fold', '2nd Fold', '3rd Fold', '4th Fold', '5th Fold'])

    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    plt.legend(loc=4)


    plt.savefig("PCG_CrossValidation.png")
    # plt.show()
    plt.close()
PCG()