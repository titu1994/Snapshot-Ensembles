import json
import numpy as np
import argparse
import sklearn.metrics as metrics
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from models import wide_residual_net as wrn

from keras.layers import Input
from keras.models import Model
from keras.datasets import cifar10
import keras.utils.np_utils as kutils

parser = argparse.ArgumentParser(description='CIFAR 10 Ensemble Prediction')

parser.add_argument('--optimize', type=int, default=0, help='Optimization flag. Set to 1 to perform a randomized '
                                                            'search to maximise classification accuracy')

parser.add_argument('--num_tests', type=int, default=20, help='Number of tests to perform when optimizing the '
                                                              'ensemble weights for maximizing classification accuracy')

args = parser.parse_args()

# Change NUM_TESTS to larger numbers to get possibly better results
NUM_TESTS = args.num_tests

# Change to False to only predict
OPTIMIZE = args.optimize

models_filenames = [r"weights/WRN-CIFAR10-16-4-Best.h5",
                    r"weights/WRN-CIFAR10-16-4-1.h5",
                    r"weights/WRN-CIFAR10-16-4-2.h5",
                    r"weights/WRN-CIFAR10-16-4-3.h5",
                    r"weights/WRN-CIFAR10-16-4-4.h5",
                    r"weights/WRN-CIFAR10-16-4-5.h5"]

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

best_acc = 0.0
best_weights = None

preds = []
for fn in models_filenames:
    model.load_weights(fn)
    yPreds = model.predict(testX, batch_size=128)
    preds.append(yPreds)

    print("Obtained predictions from model with weights = %s" % (fn))

if OPTIMIZE == 0:
    with open('models/Ensemble weights CIFAR 10.json', mode='r') as f:
        dictionary = json.load(f)

    prediction_weights = dictionary['best_weights']

    weighted_predictions = np.zeros((testX.shape[0], nb_classes), dtype='float32')
    for weight, prediction in zip(prediction_weights, preds):
        weighted_predictions += weight * prediction

    yPred = np.argmax(weighted_predictions, axis=1)
    yTrue = testY

    accuracy = metrics.accuracy_score(yTrue, yPred) * 100
    error = 100 - accuracy
    print("Accuracy : ", accuracy)
    print("Error : ", error)

    exit()

''' OPTIMIZATION REGION '''

print()

def log_loss_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_prediction = np.zeros((testX.shape[0], nb_classes), dtype='float32')

    for weight, prediction in zip(weights, preds):
        final_prediction += weight * prediction

    return log_loss(testY_cat, final_prediction)

for iteration in range(NUM_TESTS):
    prediction_weights = np.random.random(len(models_filenames))

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
    print("Iteration %d: Accuracy : " % (iteration + 1), accuracy)
    print("Iteration %d: Error : " % (iteration + 1), error)

    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = weights

    print()

print("Best Accuracy : ", best_acc)
print("Best Weights : ", best_weights)

with open('models/Ensemble weights CIFAR 10.json', mode='w') as f:
    dictionary = {'best_weights' : best_weights.tolist()}
    json.dump(dictionary, f)

''' END OF OPTIMIZATION REGION '''

