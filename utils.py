import re
import string
from collections import Counter
import numpy as np

def train_val_test_split(inputs, labels, split_frac, random_seed=None):
    
    assert len(inputs) == len(labels)
    # random shuffle data
    if random_seed:
        np.random.seed(random_seed)
    shuf_idx = np.random.permutation(len(inputs))
    inputs_shuf = np.array(inputs)[shuf_idx] 
    labels_shuf = np.array(labels)[shuf_idx]

    #make splits
    split_idx = int(len(inputs_shuf)*split_frac)
    train_x, val_x = inputs_shuf[:split_idx], inputs_shuf[split_idx:]
    train_y, val_y = labels_shuf[:split_idx], labels_shuf[split_idx:]

    test_idx = int(len(val_x)*0.5)
    val_x, test_x = val_x[:test_idx], val_x[test_idx:]
    val_y, test_y = val_y[:test_idx], val_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y


def get_batches(x, y, batch_size=100):
    
    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]