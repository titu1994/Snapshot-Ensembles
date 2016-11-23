import numpy as np
import os

import keras.callbacks as callbacks
from keras.callbacks import Callback

class SnapshotModelCheckpoint(Callback):

    def __init__(self, T, M, fn_prefix='Model'):
        super(SnapshotModelCheckpoint, self).__init__()

        self.check = T // M
        self.fn_prefix = fn_prefix

    def on_epoch_end(self, epoch, logs={}):
        if epoch != 0 and (epoch + 1) % self.check == 0:
            filepath = self.fn_prefix + "-%d.h5" % ((epoch + 1) // self.check)
            self.model.save_weights(filepath, overwrite=True)
            print("Saved snapshot at weights/%s_%d.h5" % (self.fn_prefix, epoch))


class SnapshotCallbackBuilder:

    def __init__(self, T, M, alpha_zero=0.1):
        self.T = T
        self.M = M
        self.alpha_zero = alpha_zero

    def get_callbacks(self, model_prefix='Model'):
        if not os.path.exists('weights/'):
            os.makedirs('weights/')

        callback_list = [callbacks.ModelCheckpoint("weights/%s-Best.h5" % model_prefix, monitor="val_acc",
                                                    save_best_only=True, save_weights_only=True),
                         callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule),
                         SnapshotModelCheckpoint(self.T, self.M, fn_prefix='weights/%s' % model_prefix)]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

