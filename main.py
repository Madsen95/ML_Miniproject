
from src.utils import *
from src.SVM import SVM
from src.MLP import MLP
from src.CNN import CNN

trn, trn_lbls = load_MNIST('train', p=1)
tst, tst_lbls = load_MNIST('test', p=1)

N, dim = trn.shape
print(N, dim)

do_svm = False
do_mlp = False
do_cnn = True

if __name__ == '__main__':

    #execution_times([0.1, 0.2, 0.4, 0.6, 0.8, 1])

    if do_svm:

        # SVM testing
        #svm_kernel(trn, trn_lbls, tst, tst_lbls, ['linear', 'poly', 'rbf', 'sigmoid'])
        #svm_cost_factors(trn, trn_lbls, tst, tst_lbls, ['poly', 'rbf'])
        #svm_poly_degree(trn, trn_lbls, tst, tst_lbls, [1, 2, 3, 4, 5])
        #svm_gamma_factors(trn, trn_lbls, tst, tst_lbls, [0.001, 0.01, 0.1, 1], ['poly', 'rbf'])

        # Final SVM model
        svm = SVM(trn, trn_lbls, kernel='rbf', force_train=True, save_model=True, C=10, gamma='scale')
        pred, _ = svm.make_prediction(tst)
        cm, acr = get_accuracy(pred, tst_lbls, fname='svm')
        print(acr)
        print(cm)

    if do_mlp:

        # MLP testing
        #mlp_layer_size(trn, trn_lbls, tst, tst_lbls, [10, 20, 50, 100, 200, 400, 500, 800, 1000])
        #mlp_regularization_term(trn, trn_lbls, tst, tst_lbls, [0.0001, 0.001, 0.01, 0.1, 1, 10])

        # Final MLP model
        mlp = MLP(trn, trn_lbls, force_train=True, save_model=True, layer_size=500, alpha=0.01)
        pred, _ = mlp.make_prediction(tst)
        cm, acr = get_accuracy(pred, tst_lbls, fname='mlp')
        print(acr)
        print(cm)

    if do_cnn:
        # CNN
        cnn = CNN(trn, trn_lbls, force_train=True, save_model=True)
        pred, _ = cnn.make_prediction(tst)
        cm, acr = get_accuracy(pred, tst_lbls, fname='cnn')
        print(acr)
        print(cm)