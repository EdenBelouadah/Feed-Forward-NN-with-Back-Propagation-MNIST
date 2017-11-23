import gzip,sys
import numpy as np
import pickle

"""
MiniNN - Minimal Neural Network
This code is a straigthforward and minimal implementation 
of a multi-layer neural network for training on MNIST dataset.
It is mainly intended for educational and prototyping purpuses.
"""
__author__ = "Gaetan Marceau Caron (gaetan.marceau-caron@inria.fr)"
__copyright__ = "Copyright (C) 2015 Gaetan Marceau Caron"
__license__ = "CeCILL 2.1"
__version__ = "1.0"

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def load_mnist():
    f = gzip.open('mnist.pkl.gz', 'rb')
    if(sys.version_info.major==2):
        train_set, valid_set, test_set = pickle.load(f) # compatibility issue between python 2.7 and 3.4
    else:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin-1') # compatibility issue between python 2.7 and 3.4
    f.close()

    # Shuffle
    train_X = train_set[0]
    train_y = train_set[1]
    valid_X = valid_set[0]
    valid_y = valid_set[1]

    train_perm = np.random.permutation(train_X.shape[0])
    train_set = [train_X[train_perm,:],train_y[train_perm]]

    valid_perm = np.random.permutation(valid_X.shape[0])
    valid_set = [valid_X[valid_perm,:],valid_y[valid_perm]]

    return train_set, valid_set, test_set
