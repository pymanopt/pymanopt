import random

import autograd.numpy as anp
import matplotlib
import numpy as np
import pytest
import tensorflow as tf
import torch


matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def initialize_test_state():
    seed = 42
    random.seed(seed)
    anp.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
