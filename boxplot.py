# ==============================================================================
# sinc(i) - http://sinc.unl.edu.ar/
# L. Bugnon, C. Yones, J. Raad, M. Gerard, M. Rubiolo, G. Merino, M. Pividori,
# L. Di Persia, D.H. Milone, and G. Stegmayer.
# lbugnon@sinc.unl.edu.ar
# ==============================================================================
import pandas as pd
from matplotlib import pyplot as plt

res_dir = "results/"

plt.figure("DL4Papers with FastText")

for k, entity in enumerate(["gene", "drug"]):
    plt.subplot(1, 2, k+1)
    res = pd.read_csv("%stest_%s_mutation.log" % (res_dir, entity), sep="\t")
    plt.boxplot([res["test_sensitivity"], res["test_precision"], res["test_F1"]])
    plt.ylim([0.4, 1])
    plt.xticks([1, 2, 3], ["Sensitivity", "Precision", "F1"])
    plt.title("mutation-%s" % entity)
plt.savefig('DL4pboxplot.png')
plt.show()
