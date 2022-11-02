import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Stop printing warnings
from tensorflow import keras

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
            self.model = f'data/CNN_{self.dim}dim_{self.N}trn_{self.epochs}ep_{self.batch_size}bs.h5'
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
        t0 = time.time()

        # self.clf = keras.Sequential([keras.layers.Input(shape =self.input_shape),
        #                   keras.layers.Conv2D(6, 5, padding = "same", activation = "relu"),
        #                   keras.layers.AveragePooling2D(2),
        #                   keras.layers.Conv2D(16, 5, activation = "relu"),
        #                   keras.layers.AveragePooling2D(2),
        #                   keras.layers.Conv2D(120, 5, activation = "relu"),
        #                   keras.layers.Flatten(),
        #                   keras.layers.Dense(84, activation = "relu"),
        #                   keras.layers.Dense(self.num_classes, "softmax")],name="LeNet5")
        
        self.clf = keras.Sequential()
        self.clf.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        self.clf.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        self.clf.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        self.clf.add(keras.layers.Dropout(0.25))
        self.clf.add(keras.layers.Flatten())
        self.clf.add(keras.layers.Dense(128, activation='relu'))
        self.clf.add(keras.layers.Dropout(0.5))
        self.clf.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        self.clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.history = self.clf.fit(self.x_trn, self.y_trn, batch_size=self.batch_size, epochs=self.epochs)

        t1 = time.time()
        td = t1 - t0
        print(f'Model was trained in {np.round(td, 2)}')

        self.model_summary()

        self.model = f'data/CNN_{self.dim}dim_{self.N}trn_{self.epochs}ep_{self.batch_size}bs.h5'

        self.clf.save(self.model)

        return td

    def load_model(self):
        """
        Load existing MLP model
        """
        self.clf = keras.models.load_model(self.model)
    
    def model_summary(self):

        self.clf.summary()

        if self.history is None:
            print('Cannot evaluate training for loaded model')
            return
        else:
            trn_loss = np.array(self.history.history['loss'])
            trn_acc = np.array(self.history.history['accuracy'])

            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            ax1.plot(np.arange(self.epochs)+1, trn_loss, 'b-', label = 'Loss')
            ax2.plot(np.arange(self.epochs)+1, 100*trn_acc, 'g-', label = 'Acc.')

            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Training Loss')
            ax2.set_ylabel('Training Accuracy')

            fig.legend()
    
    def make_prediction(self, tst):
        """
        Make prediction based on MLP model

        Returns:
            pred (np.array): Predictions
            td (float): Execution time [s]
        """

        N, dim = tst.shape
        if dim != self.dim:
            print('Input data wrong dimensions.')
            return

        x_tst = tst.reshape(N, int(np.sqrt(dim)), int(np.sqrt(dim)), 1)

        t0 = time.time()
        pred = np.argmax(self.clf.predict(x_tst), axis=1)
        t1 = time.time()
        td = t1 - t0

        print(f'Predictions took {np.round(td, 2)} sec')

        return pred, td