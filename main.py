
from src.utils import *
from src.SVM import SVM
from src.MLP import MLP
from src.CNN import CNN

trn, trn_lbls = load_MNIST('train', p=1)
tst, tst_lbls = load_MNIST('test', p=1)

n_features = trn.shape[1]
gamma = 1 / (n_features * trn.var())
print('scale', gamma)
print('auto', 1 / n_features)

N, dim = trn.shape
print(N, dim)

do_svm = True
do_mlp = False
do_cnn = False

if do_svm:
    #svm_kernel(trn, trn_lbls, tst, tst_lbls, ['linear', 'poly', 'rbf', 'sigmoid'])
    #svm_cost_factors(trn, trn_lbls, tst, tst_lbls, ['poly', 'rbf'])
    #svm_poly_degree(trn, trn_lbls, tst, tst_lbls, [1, 2, 3, 4, 5])
    #svm_gamma_factors(trn, trn_lbls, tst, tst_lbls, [0.001, 0.01, 0.1, 1], ['poly', 'rbf'])
    None

if do_mlp:
    # MLP
    mlp = MLP(trn, trn_lbls, layer_size=10)
    pred, _ = mlp.make_prediction(tst)
    print(get_accuracy(pred, tst_lbls))

if do_cnn:
    # CNN
    cnn = CNN(trn, trn_lbls)
    pred, _ = cnn.make_prediction(tst)
    print(get_accuracy(pred, tst_lbls))