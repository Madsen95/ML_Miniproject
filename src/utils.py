import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

from src.SVM import SVM

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
    """
    Function to calculater confusion matrix and accuracy of prediction.

    Args:
    Returns:
    """
    cm = confusion_matrix(tst_lbls, tst_pred)

    acc = np.round(np.sum(np.diagonal(cm)) / np.sum(cm) * 100, 2)

    return cm, acc

def svm_kernel_test(trn, trn_lbls, tst, tst_lbls):
    print('Testing linear kernel')
    svm = SVM(trn, trn_lbls, kernel='linear', force_train=True, save_model=False)
    pred, _ = svm.make_prediction(tst)
    cm, acc = get_accuracy(pred, tst_lbls)
    print(acc)

    plt.subplots()
    sb.heatmap(cm, annot=True)
    plt.show()
    
    print('Testing poly kernel')
    svm.train_model(kernel='poly')
    pred, _ = svm.make_prediction(tst)
    cm, acc = get_accuracy(pred, tst_lbls)
    print(acc)

    plt.subplots()
    sb.heatmap(cm, annot=True)
    plt.show()

    print('Testing rbf kernel')
    svm.train_model(kernel='rbf')
    pred, _ = svm.make_prediction(tst)
    cm, acc = get_accuracy(pred, tst_lbls)
    print(acc)

    plt.subplots()
    sb.heatmap(cm, annot=True)
    plt.show()

    print('Testing sigmoid kernel')
    svm.train_model(kernel='sigmoid')
    pred, _ = svm.make_prediction(tst)
    cm, acc = get_accuracy(pred, tst_lbls)
    print(acc)

    plt.subplots()
    sb.heatmap(cm, annot=True)
    plt.show()

def svm_cost_factors(trn, trn_lbls, tst, tst_lbls, kernel):
    acc = []
    C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    for c in C:
        print(c)
        svm = SVM(trn, trn_lbls, kernel=kernel, force_train=True, save_model=False, C=c)
        pred, _ = svm.make_prediction(tst)
        _, acr = get_accuracy(pred, tst_lbls)
        acc.append(acr)
        print(acc)

    plt.subplots()
    plt.semilogx(C, acc)
    plt.grid()
    plt.xlabel('Cost parameter C')
    plt.ylabel('Accuracy')
    plt.show()