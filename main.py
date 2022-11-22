
from src.utils import *
from src.SVM import SVM
from src.MLP import MLP
from src.CNN import CNN

trn, trn_lbls = load_MNIST('train', p=1)
tst, tst_lbls = load_MNIST('test', p=1)

N, dim = trn.shape
print(N, dim)

do_svm = True
do_mlp = False
do_cnn = False

if do_svm:

    #svm_kernel_test(trn, trn_lbls, tst, tst_lbls)

    svm_cost_factors(trn, trn_lbls, tst, tst_lbls, 'poly')
    svm_cost_factors(trn, trn_lbls, tst, tst_lbls, 'rbf')
    #svm = SVM(trn, trn_lbls, kernel='linear', force_train=False, save_model=False)
    

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