import numpy as np

import utils
mat = np.loadtxt('karate.txt')
n = np.size(mat, 0)
bf = utils.batch_feeder('../data/karate.txt',mode='cum_graph', batch_size=256, window_size=10)
context, target = next(bf)
print(context, target)

from deepwalk.skipgram import SkipGram

model = SkipGram(n, 3, 256, 10, 1.00)
model.build_graph()
embedding = model.train_model(bf, 10000, 100)
print(embedding)
np.save('embed',embedding)