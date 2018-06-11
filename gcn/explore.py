import tensorflow as tf
from utils import *

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data('cora')

