import numpy as np
from sklearn.neural_network import MLPClassifier as MLPC
import os
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

    def __init__(self, trn, trn_lbls, model=None, layer_size=None, solver='sgd', max_iter=1000):

        self.trn = trn
        self.trn_lbls = trn_lbls
        self.model = model
        self.layer_size = layer_size
        self.solver = solver
        self.max_iter = max_iter

        self.N, self.dim = self.trn.shape
        self.clf = None

        if layer_size is None:
            self.layer_size = len(np.unique(trn_lbls))

        if self.model is None:
            self.model = f'data/MLP_{self.dim}dim_{self.N}trn.joblib'
            print(f'Looking for model {self.model}')
            if os.path.isfile(self.model):
                print('Model found and loaded')
                self.load_model()
            else:
                print('No model found, training new')
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
        self.clf = MLPC(hidden_layer_sizes=self.layer_size,
                        max_iter=1000,
                        solver=self.solver,
                        random_state=1)
        self.clf.fit(self.trn, self.trn_lbls)
        t1 = time.time()
        td = t1 - t0

        print(f'Model was trained in {np.round(td, 2)} sec')
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
        pred = self.clf.predict(tst)
        t1 = time.time()
        td = t1 - t0

        print(f'Predictions took {np.round(td, 2)} sec')

        return pred, td