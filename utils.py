import numpy as np
import tensorflow as tf
from conf import *


snapshots = config_map["warmup"]
epochs    = config_map["epochs"]
lr_       = config_map["learning_rate"]


def schedule(epoch):
    cos_inner = np.pi * (epoch % (epochs // snapshots))
    cos_inner /= epochs // snapshots
    cos_out = np.cos(cos_inner) + 1
    lr = float( lr_/ 2 * cos_out)
    tf.summary.scalar('learning rate', data=lr, step=epoch)
    return lr


losses  = {"aux1": "binary_crossentropy","aux2": "binary_crossentropy",
           "aux3": "binary_crossentropy","dom" : "binary_crossentropy"
          }
weights = {"aux1": 1.0, "aux2": 1.0,"aux3": 1.0,"dom": 1.0}