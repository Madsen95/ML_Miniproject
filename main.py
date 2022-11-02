
from src.utils import *
from src.SVM import SVM
from src.MLP import MLP
from src.CNN import CNN

trn, trn_lbls = load_MNIST('train', p=0.01)
tst, tst_lbls = load_MNIST('test', p=0.01)

N, dim = trn.shape
print(N, dim)

# SVM
svm = SVM(trn, trn_lbls)
pred, _ = svm.make_prediction(tst)
print(get_accuracy(pred, tst_lbls))

# MLP
mlp = MLP(trn, trn_lbls, layer_size=10)
pred, _ = mlp.make_prediction(tst)
print(get_accuracy(pred, tst_lbls))

# CNN
cnn = CNN(trn, trn_lbls)
pred, _ = cnn.make_prediction(tst)
print(get_accuracy(pred, tst_lbls))