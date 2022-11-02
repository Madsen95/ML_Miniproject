import numpy as np
import os

from tensorflow import keras
from joblib import dump, load

class CNN:

    def __init__(self, trn, trn_lbls, model=None, batch_size=256, epochs=10):

        self.N, self.dim = trn.shape
        self.num_classes = len(np.unique(trn_lbls))
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs

        self.x_trn = trn.reshape(self.N, int(np.sqrt(self.dim)), int(np.sqrt(self.dim)), 1)
        self.y_trn = keras.utils.to_categorical(trn_lbls, num_classes=self.num_classes, dtype="float32")

        self.input_shape = self.x_trn.shape[1:]
        self.clf = None
        self.history = None

        if self.model is None:
            self.model = f'data/CNN_{self.epochs}ep_{self.batch_size}bs.joblib'
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
        self.clf = keras.Sequential([keras.layers.Input(shape =self.input_shape),
                          keras.layers.Conv2D(6, 5, padding = "same", activation = "relu"),
                          keras.layers.AveragePooling2D(2),
                          keras.layers.Conv2D(16, 5, activation = "relu"),
                          keras.layers.AveragePooling2D(2),
                          keras.layers.Conv2D(120, 5, activation = "relu"),
                          keras.layers.Flatten(),
                          keras.layers.Dense(84, activation = "relu"),
                          keras.layers.Dense(self.num_classes, "softmax")],name="LeNet5")
        
        self.clf.summary()

        self.clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.history = self.clf.fit(self.x_trn, self.y_trn, batch_size=self.batch_size, epochs=self.epochs)

        self.model = f'data/CNN_{self.epochs}ep_{self.batch_size}bs.joblib'

        dump(self.clf, self.model)

    def load_model(self):
        """
        Load existing MLP model
        """
        self.clf = load(self.model)