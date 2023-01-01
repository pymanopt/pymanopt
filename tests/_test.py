import random
import unittest

import autograd.numpy as anp
import matplotlib
import numpy as np
import tensorflow as tf
import torch


def reset_random_seed(*, seed: int):
    random.seed(seed)
    anp.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        reset_random_seed(seed=42)
        matplotlib.use("Agg")
