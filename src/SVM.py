import numpy as np
from sklearn import svm
from joblib import dump, load
import time
import os

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

    def __init__(self, trn, trn_lbls, model=None, kernel='linear', force_train=False, save_model=True, C=1, degree=3):

        self.trn = trn
        self.trn_lbls = trn_lbls
        self.model = model
        self.kernel = kernel
        self.force_train = force_train
        self.save_model = save_model
        self.C = C
        self.degree = degree

        self.N, self.dim = self.trn.shape
        self.clf = None

        if self.model is None:
            self.model = f'data/SVM_{self.dim}dim_{self.N}trn.joblib'
            print(f'Looking for model {self.model}')
            if self.force_train:
                print('Force training model')
                _ = self.train_model()  
            elif os.path.isfile(self.model):
                print('Model found and loaded')
                self.load_model()
            else:
                print('No model found, training new')
                _ = self.train_model()            
        else:
            print('Loading model', self.model)
            self.load_model()
        
    def train_model(self, kernel=None, C=None, degree=None):
        """
        Train a SVM model based on class data

        Returns:
            td (float): Execution time [s]
        """
        if kernel is not None:
            self.kernel = kernel
        if C is not None:
            self.C = C
        if degree is not None:
            self.degree = degree

        t0 = time.time()
        self.clf = svm.SVC(kernel=self.kernel, C=self.C, degree=self.degree)
        self.clf.fit(self.trn, self.trn_lbls)
        t1 = time.time()
        td = t1 - t0

        print(f'Model was trained in {np.round(td, 2)} sec')
        if self.save_model:
            dump(self.clf, f'data/SVM_{self.dim}dim_{self.N}trn.joblib')
            print('Model saved.')

        return td

    def load_model(self):
        """
        Load existing SVM model
        """
        self.clf = load(self.model)
    
    def make_prediction(self, tst):
        """
        Make prediction based on SVM model

        Returns:
            pred (np.array): Predictions
            td (float): Execution time [s]
        """
        t0 = time.time()
        pred = self.clf.predict(tst)
        t1 = time.time()
        td = t1 - t0

        print(f'Predictions took {np.round(td, 2)} sec')

        return pred, td