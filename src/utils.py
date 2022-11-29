import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import time

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
    Get confusion matrix and calculate overall classification accuracy.
    If fname is set, the confusion matrix will be plotted.

    Args:
        tst_pred (np.array): Predicted labels
        tst_lbls (np.array): True labels
        fname (string): Filename for saved plot
    Returns:
        cm (np.array): Confusion matrix
        acc (float): Overall classification accuracy
    """
    cm = confusion_matrix(tst_lbls, tst_pred)

    acc = np.round(np.sum(np.diagonal(cm)) / np.sum(cm) * 100, 2)

    if fname is not None:
        plt.subplots()
        sb.heatmap(cm, annot=True, fmt='g')
        plt.savefig(f'data/confusion_matrix_{fname}.pdf', dpi=100)

    return cm, acc

## Test functions for SVM
def svm_kernel(trn, trn_lbls, tst, tst_lbls, kernels):
    """
    SVM test function: Try different kernels and plot confusion matrices.

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        kernels (list): List of strings for kernels
    """
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
    """
    SVM test function: Try different cost parameters for choosen kernels.

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        kernels (list): List of strings for kernels
    """
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
    """
    SVM test function: Try different polynomia degrees

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        kernels (list): List of strings for kernels
    """
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
    """
    SVM test function: Try different gamma parameters.

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        gammas (np.array): Gammas to test
        kernels (list): List of strings for kernels
    """
    
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
    """
    MLP test function: Try different sizes of hidden nodes

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        nodes (list): List of nodes to test
    """
    
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
    """
    MLP test function: Try different alphas 

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        tst (np.array): Test data
        tst_lbls (np.array): Test labels
        alphas (list): List of alphas to test
    """
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


## Common functions
def execution_times(trn, trn_lbls, sizes=[0.01, 0.05]):
    """
    Common test function: Measure the training time for MLP and SVM at different
    sizes of training sets. Plot the result, and create another figure to show
    how long prediction times take.

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        sizes (list): List of training set sizes to use
    """
    svm_train = []
    svm_pred = []
    mlp_train = []
    mlp_pred = []

    for i, size in enumerate(sizes):
        trn, trn_lbls = load_MNIST('train', p=size)
        print(f'\nTest {i+1}/{len(sizes)}')

        print(f'Training SVM, size = {size}')
        t0 = time.time()
        svm = SVM(trn, trn_lbls, force_train=True, save_model=False)     
        t1 = time.time()
        td = t1 - t0
        svm_train.append(td)
        print(svm_train)

        print(f'SVM prediction starting...')
        t0 = time.time()
        _, _ = svm.make_prediction(trn)
        t1 = time.time()
        td = t1 - t0
        svm_pred.append(td)
        print(svm_pred)

        print(f'Training MLP, size = {size}')
        t0 = time.time()
        mlp = MLP(trn, trn_lbls, force_train=True, save_model=False) 
        t1 = time.time()
        td = t1 - t0
        mlp_train.append(td)
        print(mlp_train)

        print(f'MLP prediction starting...')
        t0 = time.time()
        _, _ = mlp.make_prediction(trn)
        t1 = time.time()
        td = t1 - t0
        mlp_pred.append(td)
        print(mlp_pred)

    fig, ax = plt.subplots(2, sharex=True)
    ax[0].set_title('Training time')
    ax[0].grid()
    ax[0].plot((np.array(sizes)*100).astype('str'), svm_train, c='C0')
    ax0 = ax[0].twinx()
    ax0.plot((np.array(sizes)*100).astype('str'), mlp_train, c='C1')
    ax[0].set_ylabel('SVM training time', c='C0')
    ax0.set_ylabel('MLP training time', c='C1')

    ax[1].set_title('Prediction time')
    ax[1].grid()
    ax[1].plot((np.array(sizes)*100).astype('str'), svm_pred, c='C0')
    ax1 = ax[1].twinx()
    ax1.plot((np.array(sizes)*100).astype('str'), mlp_pred, c='C1')
    ax[1].set_ylabel('SVM prediction time', c='C0')
    ax1.set_ylabel('MLP prediction time', c='C1')

    ax[1].set_xlabel('Percentage of training data')
    fig.tight_layout()

    fig.savefig('execution_times.pdf', dpi=100)