from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
from collections import namedtuple

# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
# NOTE: used as the return format for Env(), and for format to push into replay memory for off-policy methods
# NOTE: when return from Env(), state0 is always None
Experience          = namedtuple('Experience','state0, action, reward, state1, terminal1')
# NOTE: also used for on-policy methods for collect experiences over a rollout of an episode
# NOTE: policy_vb & value0_vb for storing output Variables along a rollout # NOTE: they should not be detached from the graph!
AugmentedExperience = namedtuple('AugmentedExperience', 'state0, action, one_hot_action, total_reward, inverse_out, forward_out, vec_state1, state1, terminal1, policy_vb, sigmoid_vb, value0_vb')


def loggerConfig(log_file, verbose=2):
   logger      = logging.getLogger()
   formatter   = logging.Formatter('[%(levelname)-8s] (%(processName)-11s) %(message)s')
   fileHandler = logging.FileHandler(log_file, 'w')
   fileHandler.setFormatter(formatter)
   logger.addHandler(fileHandler)
   if verbose >= 2:
       logger.setLevel(logging.DEBUG)
   elif verbose >= 1:
       logger.setLevel(logging.INFO)
   else:
       # NOTE: we currently use this level to log to get rid of visdom's info printouts
       logger.setLevel(logging.WARNING)
   return logger

def rgb2y(im):
        """Converts an RGB image to a Y image (as in YUV).
        These coefficients are taken from the torch/image library.
        Beware: these are more critical than you might think, as the
        monochromatic contrast can be surprisingly low.
        """
        if len(im.shape) < 3:
            return im
        return np.sum(im * [0.299, 0.587, 0.114], axis=2)
