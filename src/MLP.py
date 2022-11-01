import numpy as np
import pickle
from joblib import dump, load
import time

class MLP:
    """
    Support Vector Machine Class

    Args:
        trn (np.array): Training data
        trn_lbls (np.array): Training labels
        model (string): Path to saved model
        kernel (string): Kernel to use
        save (bool): Save model if true
    """

    def __init__(self, trn, trn_lbls, model=None, save=False):

        self.trn = trn
        self.trn_lbls = trn_lbls
        self.model = model
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
        """
        Train a SVM model based on class data

        Returns:
            td (float): Execution time [s]
        """
        t0 = time.time()
        # ...
        t1 = time.time()
        td = t1 - t0

        print(f'Model was trained in {np.round(td, 2)} sec')

        if self.save:
            dump(self.clf, f'data/MLP_{self.dim}dim_{self.N}trn.joblib')
            print('Model saved.')

        return td

    def load_model(self):
        """
        Load existing MLP model
        """
        self.clf = load(self.model)
    
    def make_prediction(self, tst):
        """
        Make prediction based on MLP model

        Returns:
            pred (np.array): Predictions
            td (float): Execution time [s]
        """
        t0 = time.time()
        pred = np.zeros(len(tst))
        t1 = time.time()
        td = t1 - t0

        print(f'Predictions took {np.round(td, 2)} sec')

        return pred, td