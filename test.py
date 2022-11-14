import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("subopfb.txt")

plt.plot(data[2*1024:3*1024-20])
plt.show()