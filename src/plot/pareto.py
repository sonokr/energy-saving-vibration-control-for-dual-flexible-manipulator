import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv(
    "data/2020-12-03/te0.9/se90/nsga2/gauss_n6/6/func2/data/0_data_nsga2_gauss_n6_te0.9_se90.csv"
)

X = df.iloc[:, 6]
Y = df.iloc[:, 7]
Z = df.iloc[:, 8]

fig = plt.figure()
ax = Axes3D(fig)

# 軸にラベルを付けたいときは書く
ax.set_xlabel("Link1")
ax.set_ylabel("Link2")
ax.set_zlabel("Energy")

ax.plot(X, Y, Z, marker="o", linestyle="None")

plt.show()
# fig.savefig("pareto.png", dpi=600)
