import json
import numpy as np
import sklearn.metrics as metrics

import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from snapshot import SnapshotCallbackBuilder
from models import wide_residual_net as wrn

''' Snapshot major parameters '''
M = 5 # number of snapshots
nb_epoch = T = 200 # number of epochs
alpha_zero = 0.1 # initial learning rate

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

batch_size = 128
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar10.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=False)

generator.fit(trainX, seed=0, augment=True)

init = Input(shape=(3, img_rows, img_cols))
wrn_model = wrn.create_wide_residual_network(init, nb_classes=10, N=2, k=4, dropout=0.00)
model = Model(input=init, output=wrn_model)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
print("Finished compiling")

hist = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
                   callbacks=snapshot.get_callbacks(model_prefix='WRN-CIFAR10-16-4'), # Build snapshot callbacks
                   validation_data=(testX, testY),
                   nb_val_samples=testX.shape[0])

with open('WRN-CIDAR10-16-4 training.json', mode='w') as f:
    json.dump(hist.history, f)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
