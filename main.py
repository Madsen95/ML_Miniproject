
from src.utils import *
from src.SVM import SVM
from src.MLP import MLP

trn, trn_lbls = load_MNIST('train', p=0.1)
tst, tst_lbls = load_MNIST('test', p=0.1)

N, dim = trn.shape

# SVM
svm = SVM(trn, trn_lbls)
#model = f'data/SVM_{dim}dim_{N}trn.joblib'
#svm = SVM(trn, trn_lbls, model=model)
pred, _ = svm.make_prediction(tst)
print(get_accuracy(pred, tst_lbls))

# MLP
mlp = MLP(trn, trn_lbls)