
from src.utils import *
from src.SVM import SVM

trn, trn_lbls = load_MNIST('train', p=0.1)
tst, tst_lbls = load_MNIST('test', p=0.1)

N, dim = trn.shape

# SVM
model = f'data/SVM_{dim}dim_{N}trn.joblib'
svm = SVM(trn, trn_lbls)
#svm = SVM(trn, trn_lbls, model=model)
pred, _ = svm.make_prediction(tst)
print(get_accuracy(pred, tst_lbls))