"""General configuration class for dtypes."""

from __future__ import absolute_import, print_function, division
import numpy as np
import torch

class Configuration(object):
    """Configuration class."""

    def __init__(self):
        """Initialization."""
        super(Configuration, self).__init__()

        # Dtype for computations
        self.dtype = np.float64
        self.ptdtype = torch.float64
        self.dpi = 100

        # Batch size for stability verification
        self.batch_size = 32

    @property
    def np_dtype(self):
        """Return the numpy dtype."""
        return self.dtype
