import random

import autograd.numpy as anp
import matplotlib
import numpy as np
import tensorflow as tf
import torch


def test_setup():
    seed = 42
    random.seed(seed)
    anp.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

    matplotlib.use("Agg")


test_setup()
