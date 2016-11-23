# Snapshot Ensembles in Keras

Implementation of the paper [Snapshot Ensembles: Train 1, Get M for Free](http://openreview.net/pdf?id=BJYwwY9ll) in Keras 1.1.1

# Explanation 

Snapshot Ensemble is a method to obtain multiple neural network which can be ensembled at no additional training cost. This is achieved by letting a single neural network converge into several local minima along its optimization path and save the model parameters at certain epochs, therefore the weights being "snapshots" of the model. 

The repeated rapid convergence is realized using cosine annealing cycles as the learning rate schedule. It can be described by:<br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/cosine%20annealing%20schedule.JPG?raw=true' width=50%>

This scheduler provides a learning rate which is similar to the below image. Note that the learning rate never actually becomes 0, it just gets very close to it (~0.0005): <br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/cosing%20wave.png?raw=true' width=75% height=75%>

The theory behind using a learning rate schedule which occilates between such extreme values (0.1 to 5e-4, M times) is that there exist multiple local minima when training a model. Constantly reducing the local learning rate can force the model to be stuck at a less than optimal local minima. Therefore, to escape, we use a very large learning rate to escape the current local minima and attempt to find another possibly better local minima.

It can be properly described using the following image:<br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/local%20minima.JPG?raw=true'>

Figure 1: Left: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training. Right: Illustration of Snapshot Ensembling optimization. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima. We take a snapshot at each minimum for test time ensembling.

# Usage

The paper uses several models such as ResNet-101, Wide Residual Network and DenseNet-40 and DenseNet-100. While DenseNets are the highest performing models in the paper, they are too large and take extremely long to train. Therefore, the current trained model is the Wide Residual Net (16-4) setting. This model performs poorly compared to the 34-4 version but trains several times faster.

The technique is simple to implement in Keras, using a custom callback. These callbacks can be built using the `SnapshotCallbackBuilder` class in `snapshot.py`. Other models can simply use this callback builder to other models to train them in a similar manner.

To use snapshot ensemble in other models : 
```
from snapshot import SnapshotCallbackBuilder

M = 5 # number of snapshots
nb_epoch = T = 200 # number of epochs
alpha_zero = 0.1 # initial learning rate
model_prefix = 'Model_'

snapshot = SnapshotCallbackBuilder(T, M, alpha_zero) 
...
model = Sequential() OR model = Model(ip, output) # Some model that has been compiled

model.fit(trainX, trainY, callbacks=snapshot.get_callbacks(model_prefix=model_prefix))
```

To train WRN or DenseNet models on CIFAR 10 or 100 (or use pre trained models):

1. Download the 6 WRN-16-4 weights that are provided in the Release tab of the project and place them in the `weights` directory for CIFAR 10 or 100
2. Run the `train_cifar_10.py` script to train the WRN-16-4 model on CIFAR-10 dataset (not required since weights are provided)
3. Run the `predict_cifar_10.py` script to make an ensemble prediction. 

Note the difference on calculating only the predictions of the best model (92.70 % accuracy), and the weighted ensemble version of the Snapshots (92.84 % accuracy). The difference is minor, but still an improvement. 

The improvement is minor due to the fact that the model is far smaller than the WRN-34-4 model, nor is it trained on the CIFAR-100 or Tiny ImageNet dataset. According to the paper, models trained on more complex datasets such as CIFAR 100 and Tiny ImageNet obtaines a greater boost from the ensemble model.

## Parameters
Some parameters for WRN models from the paper:
- M = 5
- nb_epoch = 200
- alpha_zero = 0.1
- wrn_N = 2 (WRN-16-4) or 4 (WRN-28-8)
- wrn_k = 4 (WRN-16-4) or 8 (WRN-28-8)

Some parameters for DenseNet models from the paper:
- M = 6
- nb_epoch = 300
- alpha_zero = 0.2
- dn_depth = 40 (DenseNet-40-12) or 100 (DenseNet-100-24)
- dn_growth_rate = 12 (DenseNet-40-12) or 24 (DenseNet-100-24)

### train_*.py
```
--M              : Number of snapshots that will be taken. Optimal range is in between 4 - 8. Default is 5
--nb_epoch       : Number of epochs to train the network. Default is 200
--alpha_zero     : Initial Learning Rate. Usually 0.1 or 0.2. Default is 0.1

--model          : Type of model to train. Can be "wrn" for Wide ResNets or "dn" for DenseNet

--wrn_N          : Number of WRN blocks. Computed as N = (n - 4) / 6. Default is 2.
--wrn_k          : Width factor of WRN. Default is 12.

--dn_depth       : Depth of DenseNet. Default is 40.
--dn_growth_rate : Growth rate of DenseNet. Default is 12.
```

### predict_*.py
```
--optimize       : Flag to optimize the ensemble weights. 
                   Default is 0 (Predict using optimized weights).
                   Set to 1 to optimize ensemble weights (test for num_tests times).
                   Set to -1 to predict using equal weights for all models (As given in the paper).
               
--num_tests      : Number of times the optimizations will be performed. Default is 20

--model          : Type of model to train. Can be "wrn" for Wide ResNets or "dn" for DenseNet

--wrn_N          : Number of WRN blocks. Computed as N = (n - 4) / 6. Default is 2.
--wrn_k          : Width factor of WRN. Default is 12.

--dn_depth       : Depth of DenseNet. Default is 40.
--dn_growth_rate : Growth rate of DenseNet. Default is 12.
```

# Performance
- Single Best: Describes the performance of the single best model.
- Without Optimization: Describes the performance of the ensemble model with equal weights for all models
- With Optimization: Describes the performance of the ensemble model with optimized weights found via minimization of log-loss scores

<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/classification_scores.JPG?raw=true' width=100%>

# Requirements

- Keras
- Theano (tested) / Tensorflow (not tested, weights not available but can be converted)
- scipy
- h5py
- sklearn
