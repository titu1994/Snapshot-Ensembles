import numpy as np
import sklearn.metrics as metrics
from models import wide_residual_net as wrn

from keras.layers import Input
from keras.models import Model
from keras.datasets import cifar100
import keras.utils.np_utils as kutils

models_filenames = [r"weights/WRN-CIFAR100-16-4-Best.h5",
                    r"weights/WRN-CIFAR100-16-4-1.h5",
                    r"weights/WRN-CIFAR100-16-4-2.h5",
                    r"weights/WRN-CIFAR100-16-4-3.h5",
                    r"weights/WRN-CIFAR100-16-4-4.h5",
                    r"weights/WRN-CIFAR100-16-4-5.h5"]

prediction_weights = [1] * 6

prediction_weights[0] = 2 # Increase weight of best model against other models

(trainX, trainY), (testX, testY) = cifar100.load_data()

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY = kutils.to_categorical(trainY)

init = Input(shape=(3, 32, 32))
wrn_model = wrn.create_wide_residual_network(init, nb_classes=100, N=2, k=4, dropout=0.00)
model = Model(input=init, output=wrn_model)

preds = np.zeros((testX.shape[0], 100), dtype='float32')

for fn, prediction_weight in zip(models_filenames, prediction_weights):
    model.load_weights(fn)
    yPreds = model.predict(testX)
    preds += yPreds * prediction_weight

    print("Obtained predictions from model with weights = %s" % fn)

preds /= sum(prediction_weights)

yPred = np.argmax(preds, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)