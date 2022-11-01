import numpy as np
from sklearn import svm
import pickle
from joblib import dump, load
import time

class SVM:
    """
    Support Vector Machine Class

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        model (string): Path to saved model
        kernel (string): Kernel to use
        save (bool): Save model if true
    """

    def __init__(self, trn, trn_lbls, model=None, kernel='linear', save=False):

        self.trn = trn
        self.trn_lbls = trn_lbls
        self.model = model
        self.kernel = kernel
        self.save = save

        self.N, self.dim = self.trn.shape
        self.clf = None

        if self.model is None:
            print('Training new model')
            _ = self.train_model()
            
        else:
            print('Loading model', self.model)
            self.load_model()
        
    def train_model(self):
        t0 = time.time()
        self.clf = svm.SVC(kernel=self.kernel)
        self.clf.fit(self.trn, self.trn_lbls)
        t1 = time.time()
        td = t1 - t0

        print(f'Model was trained in {np.round(td, 2)} sec')

        if self.save:
            dump(self.clf, f'data/SVM_{self.dim}dim_{self.N}trn.joblib')
            print('Model saved.')

        return td

    def load_model(self):
        self.clf = load(self.model)
    
    def make_prediction(self, tst):
        t0 = time.time()
        pred = self.clf.predict(tst)
        t1 = time.time()
        td = t1 - t0

        print(f'Predictions took {np.round(td, 2)} sec')

        return pred, td