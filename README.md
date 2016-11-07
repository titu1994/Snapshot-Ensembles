# Snapshot Ensembles in Keras

Implementation of the paper [Snapshot Ensembles: Train 1, Get M for Free](http://openreview.net/pdf?id=BJYwwY9ll) in Keras 1.1.1

# Explanation 

Snapshot Ensemble is a method to obtain multiple neural network which can be ensembled at no additional training cost. This is achieved by letting a single neural network converge into several local minima along its optimization path and save the model parameters at certain epochs, therefore the weights being "snapshots" of the model. 

The repeated rapid convergence is realized using cosine annealing cycles as the learning rate schedule. It can be described by:<br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/cosine%20annealing%20schedule.JPG?raw=true' width=50%>

This scheduler provides a learning rate which is similar to the below image. Note that the learning rate never actually becomes 0, it just gets very close to it (~0.0005): <br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/cosing%20wave.png?raw=true' width=75% height=75%>

The theory behind using a learning rate schedule which occilates between such extreme values (0.1 to 1e-3 in M iterations) is that there exist multiple local minima when training a model. Constantly reducing the local learning rate can force the model to be stuck at a less than optimal local minima. To escape, we attempt to use a sudden, very large learning rate to escape the current local minima and attempt to find another one.

It can be properly described using the following image:<br>
<img src='https://github.com/titu1994/Snapshot-Ensembles/blob/master/images/local%20minima.JPG?raw=true'>

Figure 1: Left: Illustration of SGD optimization with a typical learning rate schedule. The model converges to a minimum at the end of training. Right: Illustration of Snapshot Ensembling optimization. The model undergoes several learning rate annealing cycles, converging to and escaping from multiple local minima. We take a snapshot at each minimum for test time ensembling.

# Usage

The paper uses several models such as ResNet-101, Wide Residual Network and DenseNet-40 and DenseNet-100. While DenseNets are the highest performing models in the paper, they are too large and take extremely long to train. Therefore, the current trained model is the Wide Residual Net (16-4) setting. This model performs poorly compared to the 34-4 version but trains several times faster.

1. Download the 6 WRN-16-4 weights that are provided in the Release tab of the project and place them in the `weights` directory
2. Run the `train_cifar_10.py` script to train the WRN-16-4 model on CIFAR-10 dataset (not required since weights are provided)
3. Run the `predict_cifar_10.py` script to make an ensemble prediction.

Note the difference on calculating only the predictions of the best model (92.70 % accuracy), and the weighted ensemble version of the Snapshots (92.79 % accuracy). Note that you can comment the `prediction_weights[0] = 2` line, since that follows the original paper's methodology for ensemble prediction. However this reduces the accuracy of the model (92.77 %). The difference is minor, but still an improvement. 

The improvement is minor due to the fact that the model is far smaller than the WRN-34-4 model, nor is it trained on the CIFAR-100 or Tiny ImageNet dataset. According to the paper, models trained on more complex datasets such as CIFAR 100 and Tiny ImageNet obtaines a greater boost from the ensemble model.

# Requirements

- Keras 1.1.1
- Theano (tested) / Tensorflow (not tested, weights not available)
- h5py
- sklearn
