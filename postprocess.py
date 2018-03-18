import numpy as np

embedding = np.load('embed.npy')
import matplotlib.pyplot as plt

import scipy.io as scio
content = scio.loadmat('/home/drproduck/Documents/circledata_50.mat', mat_dtype=True)
gnd = content['gnd']
plt.scatter(embedding[:,0], embedding[:,1], c=gnd[:,0])
plt.show()