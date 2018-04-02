import numpy as np

embedding = np.load('embed.npy')
import matplotlib.pyplot as plt

import scipy.io as scio
# content = scio.loadmat('/home/drproduck/Documents/circledata_50.mat', mat_dtype=True)
# gnd = content['gnd']
from mpl_toolkits.mplot3d import Axes3D
gnd = np.loadtxt('../data/karatey.txt')
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=gnd)


# plt.scatter(embedding[:,0], embedding[:,1], c=gnd)
plt.show()