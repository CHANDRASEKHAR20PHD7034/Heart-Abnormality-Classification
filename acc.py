def acc():

    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    barWidth = 0.10

    ProposedCDTWSA = [95.81,    95.9]
    ExistingWSA = [94.79,    94.36]
    ExistingTBS = [93.17,    93.37]
    ExistingRGS = [92.24,    92.2]
    ExistingEBS = [ 90.42,    90.73]
    ExistingErS = [89.17,    89.23]

    br1 = np.arange(len(ProposedCDTWSA))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]


    plt.bar(br1, ProposedCDTWSA, color ='wheat',edgecolor='black',  width = barWidth,   label ='Proposed PJM-DJRNN')
    plt.bar(br2, ExistingWSA, color ='darksalmon',edgecolor='black', width = barWidth, label ='CGS-DJRNN')
    plt.bar(br3, ExistingTBS, color ='darkturquoise',edgecolor='black', width = barWidth,  label ='RGS-DJRNN')
    plt.bar(br4, ExistingRGS, color ='plum',edgecolor='black', width = barWidth,   label ='BGS-DJRNN')
    plt.bar(br5, ExistingEBS, color ='y' ,edgecolor='black',width = barWidth,  label ='HGS-DJRNN')
    plt.bar(br6, ExistingErS, color='#8A3324', edgecolor='black',width = barWidth,  label='BOA-DJRNN')
    # plt.bar(br7, ExistingEhS, color='#3D9140',edgecolor='black', width = barWidth,  label='VGG-19')
    # plt.bar(br8, ExistingElS, color='#F08080', edgecolor='black', width=barWidth, label='AlexNet')
    # plt.bar(br9, ExistingElS, color='#8B8386', edgecolor='black', width=barWidth, label='ResNet34')

    plt.title("", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([r + barWidth for r in range(len(ProposedCDTWSA))], ["Accuracy","Precision"])
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.xticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)
    plt.yticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)


    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    # plt.legend(loc=2, bbox_to_anchor=(0.50, 1))
    plt.legend(loc="lower right")
    plt.savefig("ecg.png")
    plt.show()
    plt.close()
acc()

def acc1():

    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    barWidth = 0.10

    ProposedCDTWSA = [95.31,    95.40]
    ExistingWSA = [93.86,    93.98]
    ExistingTBS = [91.37,    91.87]
    ExistingRGS = [90.12,    90.62]
    ExistingEBS = [ 89.37,    89.82]
    ExistingErS = [87.64,    87.78]

    br1 = np.arange(len(ProposedCDTWSA))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]


    plt.bar(br1, ProposedCDTWSA, color ='#F5F5F5',edgecolor='black',hatch="////" , width = barWidth,   label ='Proposed PJM-DJRNN')
    plt.bar(br2, ExistingWSA, color ='#EE82EE',edgecolor='black',hatch="////" , width = barWidth, label ='CGS-DJRNN')
    plt.bar(br3, ExistingTBS, color ='#FF6347',edgecolor='black',hatch="////" , width = barWidth,  label ='RGS-DJRNN')
    plt.bar(br4, ExistingRGS, color ='plum',edgecolor='black', hatch="////" ,width = barWidth,   label ='BGS-DJRNN')
    plt.bar(br5, ExistingEBS, color ='y' ,edgecolor='black',hatch="////" ,width = barWidth,  label ='HGS-DJRNN')
    plt.bar(br6, ExistingErS, color='#8A3324', edgecolor='black',hatch="////" ,width = barWidth,  label='BOA-DJRNN')
    # plt.bar(br7, ExistingEhS, color='#3D9140',edgecolor='black', width = barWidth,  label='VGG-19')
    # plt.bar(br8, ExistingElS, color='#F08080', edgecolor='black', width=barWidth, label='AlexNet')
    # plt.bar(br9, ExistingElS, color='#8B8386', edgecolor='black', width=barWidth, label='ResNet34')

    plt.title("", fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xlabel('Metrics', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.ylabel('Values (%)', fontweight='bold', fontname="Times New Roman", fontsize=14)
    plt.xticks([r + barWidth for r in range(len(ProposedCDTWSA))], ["Accuracy","Precision"])
    plt.rcParams['font.sans-serif'] = "Times New Roman"
    plt.xticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)
    plt.yticks(fontweight='bold',
               fontname="Times New Roman", fontsize=14)


    plt.rcParams['font.size'] = 14
    plt.rcParams['font.weight'] = 'bold'
    # plt.legend(loc=2, bbox_to_anchor=(0.50, 1))
    plt.legend(loc="lower right")
    plt.savefig("pcg.png")
    plt.show()
    plt.close()
acc1()