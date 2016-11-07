import json

import keras.callbacks as callbacks
import keras.utils.np_utils as kutils
import numpy as np
import sklearn.metrics as metrics
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from models import wide_residual_net as wrn

''' Snapshot major parameters '''
M = 5 # number of snapshots
nb_epoch = T = 200 # number of epochs
alpha_zero = 0.1 # initial learning rate

batch_size = 128
img_rows, img_cols = 32, 32


def cosine_anneal_schedule(t):
    cos_inner = np.pi * (t % (T // M)) # t - 1 is used when t has 1-based indexing.
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    return float(alpha_zero / 2 * cos_out)


class SnapshotModelCheckpoint(Callback):

    def __init__(self, T, M, fn_prefix='WRN'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = T // M
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))


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
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=True)

init = Input(shape=(3, img_rows, img_cols))
wrn_model = wrn.create_wide_residual_network(init, nb_classes=10, N=2, k=4, dropout=0.00)

model = Model(input=init, output=wrn_model)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
print("Finished compiling")

hist = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
                   callbacks=[callbacks.ModelCheckpoint("weights/WRN-CIFAR10-16-4-Best.h5", monitor="val_acc",
                                                        save_best_only=True, save_weights_only=True),
                              callbacks.LearningRateScheduler(schedule=cosine_anneal_schedule),
                              SnapshotModelCheckpoint(T, M, fn_prefix='weights/WRN-CIFAR10-16-4')],
                   validation_data=(testX, testY),
                   nb_val_samples=testX.shape[0])

with open('WRN-16-4 training.json', mode='w') as f:
    json.dump(hist.history, f)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
