import random
import unittest

import autograd.numpy as anp
import numpy as np
import tensorflow as tf
import torch


class TestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_seed()

    @staticmethod
    def _set_seed():
        seed = 42
        random.seed(seed)
        anp.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tf.random.set_seed(seed)
