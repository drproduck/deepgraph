import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import matop

content = scio.loadmat('/home/drproduck/Documents/DS2012.mat', mat_dtype=True)
fea = content['fea']
SIGMA = 100

w = matop.eudist(fea, fea, False)
w = np.exp(-w/(2*(SIGMA**2)))

import utils
bf = utils.batch_feeder(None, w, 'graph', 8, 1)
context, target = next(bf)
print(context, target)


