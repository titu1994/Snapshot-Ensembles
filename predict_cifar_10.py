import numpy as np
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as wrn

from keras.layers import Input
from keras.models import Model
from keras.datasets import cifar10
import keras.utils.np_utils as kutils

models_filenames = [r"weights/WRN-CIFAR10-16-4-Best.h5",
                    r"weights/WRN-CIFAR10-16-4-1.h5",
                    r"weights/WRN-CIFAR10-16-4-2.h5",
                    r"weights/WRN-CIFAR10-16-4-3.h5",
                    r"weights/WRN-CIFAR10-16-4-4.h5",
                    r"weights/WRN-CIFAR10-16-4-5.h5"]

prediction_weights = np.array([0.5] * len(models_filenames))

(trainX, trainY), (testX, testY) = cifar10.load_data()
nb_classes = len(np.unique(testY))

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

trainY = kutils.to_categorical(trainY)
testY_cat = kutils.to_categorical(testY)

init = Input(shape=(3, 32, 32))
wrn_model = wrn.create_wide_residual_network(init, nb_classes=10, N=2, k=4, dropout=0.00)
model = Model(input=init, output=wrn_model)

preds = []

for fn, prediction_weight in zip(models_filenames, prediction_weights):
    model.load_weights(fn)
    yPreds = model.predict(testX)
    preds.append(yPreds)

    print("Obtained predictions from model with weights = %s" % fn)


def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((testX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, preds):
            final_prediction += weight * prediction

    return log_loss(testY_cat, final_prediction)


constraints = ({'type': 'eq', 'fun':lambda w: 1 - sum(w)})
bounds = [(0, 1)] * len(preds)

result = minimize(log_loss_func, prediction_weights, method='SLSQP', bounds=bounds, constraints=constraints)

print('Best Ensemble Weights: {weights}'.format(weights=result['x']))

weights = result['x']
weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')

for weight, prediction in zip(weights, preds):
    weighted_predictions += weight * prediction

yPred = np.argmax(weighted_predictions, axis=1)
yTrue = testY

accuracy = metrics.accuracy_score(yTrue, yPred) * 100
error = 100 - accuracy
print("Accuracy : ", accuracy)
print("Error : ", error)