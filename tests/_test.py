import random
import unittest

import autograd.numpy as anp
import numpy as np
import tensorflow as tf
import torch


class TestCase(unittest.TestCase):
    @classmethod
    def testSetUp(cls):
        seed = 42
        random.seed(seed)
        anp.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tf.random.set_seed(seed)
