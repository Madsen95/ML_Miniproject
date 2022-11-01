import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

def load_MNIST(data_name, p=1, K=10, Norm=255.0):
    """
    Function to load and normalize arbitrary data from MNIST data set.

    Args:
        data_name (string): Data to load from mnist
        p (float): Percentage of dataset to use
        K (int): Number of classes to load
        Norm (float): Data normalization factor
    Returns:
        data_set (np.array): MNIST dataset
        labels (np.array): Labels for samples
    """
    mnist = loadmat("data/mnist_all.mat")
    data_set = None
    
    for k in range(K):
        temp = mnist[f'{data_name}{k}']
        data = temp[0:int(len(temp) * p), :]

        if data_set is None:
            data_set = data / Norm
            labels = np.ones(data.shape[0], dtype=int) * k
        else:
            data_set = np.concatenate([data_set, data / Norm])
            labels = np.concatenate([labels, np.ones(data.shape[0], dtype=int) * k])
    
    return data_set, labels

def get_accuracy(tst_pred, tst_lbls):
    cm = confusion_matrix(tst_lbls, tst_pred)
    return cm