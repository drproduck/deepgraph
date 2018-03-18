import numpy as np
import tensorflow as tf
import os
import scipy.io as scio
import matop

content = scio.loadmat('/home/drproduck/Documents/circledata_50.mat', mat_dtype=True)
fea = content['fea']
SIGMA = 30

w = matop.eudist(fea, fea, False)
w = np.exp(-w/(2*(SIGMA**2)))

import utils
bf = utils.batch_feeder(None, w, 'graph', 128, 20)
context, target = next(bf)
print(context, target)

from skipgram import SkipGram

model = SkipGram(np.size(fea, 0), 20, 128, 10, 1.00)
model.build_graph()
embedding = model.train_model(bf, 5000, 100)
print(embedding)
np.save('embed',embedding)
