import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_file = "../dataset_idx/mnist/cluster_full/5client/mnist_full_stat.csv"

data = np.loadtxt(csv_file, delimiter=",", dtype=np.int32)
print("Num client:", data.shape[0], "Num class:", data.shape[1])

plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams.update({'font.size': 16})
plt.rc('axes', titlesize=5)     # fontsize of the axes title
plt.rc('axes', labelsize=25)    # fontsize of the x and y labels

ax = sns.heatmap(data, annot=True, fmt="d", cbar=False, linewidths=.5, cmap="YlGnBu")
plt.xlabel("Class")
plt.ylabel("Client")

plt.savefig("./figures/mnist_cluster_full_N5_K5/mnist/cluster_full/5client/mnist_full/data_dis.png", dpi=128)
