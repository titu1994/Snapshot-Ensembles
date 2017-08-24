import json
import numpy as np
import sklearn.metrics as metrics
import argparse
import os

import keras.utils.np_utils as kutils
from keras.datasets import cifar100
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from snapshot import SnapshotCallbackBuilder
from models import wide_residual_net as WRN, dense_net as DN

parser = argparse.ArgumentParser(description='CIFAR 100 Ensemble Prediction')

parser.add_argument('--M', type=int, default=5, help='Number of snapshots')
parser.add_argument('--nb_epoch', type=int, default=200, help='Number of training epochs')
parser.add_argument('--alpha_zero', type=float, default=0.1, help='Initial learning rate')
parser.add_argument('--model', type=str, default='wrn', help='Type of model to train')

# Wide ResNet Parameters
parser.add_argument('--wrn_N', type=int, default=2, help='Number of WRN blocks. Computed as N = (n - 4) / 6.')
parser.add_argument('--wrn_k', type=int, default=4, help='Width factor of WRN')

# DenseNet Parameters
parser.add_argument('--dn_depth', type=int, default=40, help='Depth of DenseNet')
parser.add_argument('--dn_growth_rate', type=int, default=12, help='Growth rate of DenseNet')

args = parser.parse_args()

''' Snapshot major parameters '''
M = args.M # number of snapshots
nb_epoch = T = args.nb_epoch # number of epochs
alpha_zero = args.alpha_zero # initial learning rate

model_type = str(args.model).lower()
assert model_type in ['wrn', 'dn'], 'Model type must be one of "wrn" for Wide ResNets or "dn" for DenseNets'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

batch_size = 128 if model_type == "wrn" else 64
img_rows, img_cols = 32, 32

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY = kutils.to_categorical(trainY)
testY_cat = kutils.to_categorical(testY)

generator = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=False)

generator.fit(trainX, seed=0, augment=True)

if K.image_dim_ordering() == "th":
    init = (3, img_rows, img_cols)
else:
    init = (img_rows, img_cols, 3)

if not os.path.exists('weights/'):
    os.makedirs('weights/')

if model_type == "wrn":
    model = WRN.create_wide_residual_network(init, nb_classes=100, N=args.wrn_N, k=args.wrn_k, dropout=0.00)

    model_prefix = 'weights/WRN-CIFAR100-%d-%d' % (args.wrn_N * 6 + 4, args.wrn_k)
else:
    model = DN.create_dense_net(nb_classes=100, img_dim=init, depth=args.dn_depth, nb_dense_block=1,
                                growth_rate=args.dn_growth_rate, nb_filter=16, dropout_rate=0.2)

    model_prefix = 'weights/DenseNet-CIFAR100-%d-%d' % (args.dn_depth, args.dn_growth_rate)

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["acc"])
print("Finished compiling")

hist = model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
                   callbacks=snapshot.get_callbacks(model_prefix=model_prefix), # Build snapshot callbacks
                   validation_data=(testX, testY_cat),
                           nb_val_samples=testX.shape[0])

with open(model_prefix + ' training.json', mode='w') as f:
    json.dump(hist.history, f)

yPreds = model.predict(testX)
yPred = np.argmax(yPreds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)
