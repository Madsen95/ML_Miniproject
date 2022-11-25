import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt

from src.SVM import SVM
from src.MLP import MLP

## Multipurpose functions
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

def get_accuracy(tst_pred, tst_lbls, fname=None):
    """
    Function to calculater confusion matrix and accuracy of prediction.

    Args:
    Returns:
    """
    cm = confusion_matrix(tst_lbls, tst_pred)

    acc = np.round(np.sum(np.diagonal(cm)) / np.sum(cm) * 100, 2)

    if fname is not None:
        plt.subplots()
        sb.heatmap(cm, annot=True, fmt='g')
        plt.savefig(f'confusion_matrix_{fname}.pdf', dpi=100)

    return cm, acc

## Test functions for SVM
def svm_kernel(trn, trn_lbls, tst, tst_lbls, kernels):

    for kernel in kernels:

        print('Testing kernel:', kernel)
        svm = SVM(trn, trn_lbls, kernel='linear', force_train=True, save_model=False)
        pred, _ = svm.make_prediction(tst)
        cm, acc = get_accuracy(pred, tst_lbls)
        print(acc)

        plt.subplots()
        sb.heatmap(cm, annot=True)
        plt.show()

def svm_cost_factors(trn, trn_lbls, tst, tst_lbls, kernels):

    fig, ax = plt.subplots(figsize=([8.0, 6.0/2]))

    for kernel in kernels:
        acc = []
        C = [0.01, 0.1, 1, 10, 100, 1000, 10000]
        for c in C:
            print(c)
            svm = SVM(trn, trn_lbls, kernel=kernel, force_train=True, save_model=False, C=c)
            pred, _ = svm.make_prediction(tst)
            _, acr = get_accuracy(pred, tst_lbls)
            acc.append(acr)
            print(acc)

        ax.semilogx(C, acc, label=kernel)

    ax.legend()
    ax.grid()
    ax.set_xlabel('Cost parameter C')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    #fig.savefig('dev_svm_C.pdf', dpi=100)
    
def svm_poly_degree(trn, trn_lbls, tst, tst_lbls, degrees):
    acc = []

    for degree in degrees:
        print(degree)
        svm = SVM(trn, trn_lbls, kernel='poly', force_train=True, save_model=False, degree=degree)
        pred, _ = svm.make_prediction(tst)
        _, acr = get_accuracy(pred, tst_lbls)
        acc.append(acr)
        print(acc)

    fig, ax = plt.subplots(figsize=([8.0, 6.0/2]))
    ax.plot(degrees, acc)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel('Polynomial degree')
    ax.set_ylabel('Accuracy')
    ax.grid()
    fig.tight_layout()

    fig.savefig('dev_svm_poly_degree.pdf', dpi=100)

def svm_gamma_factors(trn, trn_lbls, tst, tst_lbls, gammas, kernels):
    
    fig, ax = plt.subplots(figsize=([8.0, 6.0/2]))

    for kernel in kernels:
        print(kernel)
        acc = []
        for g in gammas:
            print(g)
            svm = SVM(trn, trn_lbls, kernel=kernel, force_train=True, save_model=False, gamma=g)
            pred, _ = svm.make_prediction(tst)
            _, acr = get_accuracy(pred, tst_lbls)
            acc.append(acr)
            print(acc)

        ax.semilogx(gammas, acc, label=kernel)

        print(kernel, 'default settings')
        gammas_d = ['scale', 'auto']
        for g in gammas_d:
            print(g)
            svm = SVM(trn, trn_lbls, kernel=kernel, force_train=True, save_model=False, gamma=g)
            pred, _ = svm.make_prediction(tst)
            _, acr = get_accuracy(pred, tst_lbls)
            print(acr)

            ax.scatter(svm.clf._gamma, acr, label=f'{kernel}: {g}')

    ax.legend()
    ax.grid()
    ax.set_xlabel('Cost parameter C')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig('dev_svm_gamma.pdf', dpi=100)

## Test functions for MLP
def mlp_layer_size(trn, trn_lbls, tst, tst_lbls, nodes):
    
    acc_tst = []
    acc_trn = []
    td = []

    for node in nodes:

        mlp = MLP(trn, trn_lbls, layer_size=node, force_train=True, save_model=False)
        pred_tst, _ = mlp.make_prediction(tst)
        pred_trn, _ = mlp.make_prediction(trn)
        _, acr_tst = get_accuracy(pred_tst, tst_lbls)
        _, acr_trn = get_accuracy(pred_trn, trn_lbls)
        acc_tst.append(acr_tst)
        acc_trn.append(acr_trn)
        td.append(mlp.td)
        print(acc_tst, acc_trn)

    fig, ax = plt.subplots(figsize=([8.0, 6.0/2]))
    ax.plot(np.array(nodes).astype('str'), acc_tst, label='Test accuracy')
    ax.plot(np.array(nodes).astype('str'), acc_trn, label='Train accuracy')

    ax2 = ax.twinx()
    ax2.plot(np.array(nodes).astype('str'), td, c='C2', label='Trainig time')
    ax2.set_ylabel('Training time [s]', color='C2')

    ax.legend()
    ax.grid()
    ax.set_xlabel('Cost parameter C')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig('dev_mlp_layers.pdf', dpi=100)

def mlp_regularization_term(trn, trn_lbls, tst, tst_lbls, alphas):
    
    acc_tst = []
    acc_trn = []
    td = []

    for alpha in alphas:

        mlp = MLP(trn, trn_lbls, alpha=alpha, force_train=True, save_model=False)
        pred_tst, _ = mlp.make_prediction(tst)
        pred_trn, _ = mlp.make_prediction(trn)
        _, acr_tst = get_accuracy(pred_tst, tst_lbls)
        _, acr_trn = get_accuracy(pred_trn, trn_lbls)
        acc_tst.append(acr_tst)
        acc_trn.append(acr_trn)
        td.append(mlp.td)
        print(acc_tst, acc_trn)

    fig, ax = plt.subplots(figsize=([8.0, 6.0/2]))
    ax.plot(np.array(alphas).astype('str'), acc_tst, label='Test accuracy')
    ax.plot(np.array(alphas).astype('str'), acc_trn, label='Train accuracy')

    ax2 = ax.twinx()
    ax2.plot(np.array(alphas).astype('str'), td, c='C2', label='Trainig time')
    ax2.set_ylabel('Training time [s]', color='C2')

    ax.legend()
    ax.grid()
    ax.set_xlabel('Regularization term l2')
    ax.set_ylabel('Accuracy')
    fig.tight_layout()
    fig.savefig('dev_mlp_alphas.pdf', dpi=100)

    print('tst', acc_tst)
    print('trn', acc_trn)
    print('td', td)