from __future__ import absolute_import
from __future__ import division
import numpy as np
import os
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.helpers import loggerConfig
from utils.sharedAdam import SharedAdam

class Params(object):   # NOTE: shared across all modules
    def __init__(self):
        self.verbose     = 0            # 0(warning) | 1(info) | 2(debug)

        # training signature
        self.machine     = "desktop1"  # "machine_id"
        self.timestamp   = "17072200"   # "yymmdd##"
        self.visdom_port = 8098
        # training configuration
        self.mode        = 1            # 1(train) | 2(test model_file)

        self.continue_training = True   # Continues training if a model already exists, otherwise starts from 0

        self.seed        = 123
        self.render      = False        # whether render the window from the original envs or not
        self.visualize   = True         # whether do online plotting and stuff or not

        self.num_processes      = 3
        self.hist_len           = 4
        self.hidden_dim         = 256

        self.use_cuda           = False   # Not used
        self.dtype              = torch.FloatTensor

        # prefix for model/log/visdom
        self.refs        = self.machine + "_" + self.timestamp # NOTE: using this as env for visdom
        self.root_dir    = os.getcwd()

        # model files
        self.model_name  = self.root_dir + "/models/" + self.refs + ".pth"

        if self.continue_training:
            if os.path.exists(self.model_name):
                self.model_file  = self.model_name
            else:
                self.model_file  = None

        if self.mode == 2:
            self.model_file  = self.model_name  # NOTE: so only need to change self.mode to 2 to test the current training
            assert self.model_file is not None, "Pre-Trained model is None, Testing aborted!!!"
            self.refs = self.refs + "_test"     # NOTE: using this as env for visdom for testing, to avoid accidentally redraw on the training plots

        # logging configs
        self.log_name    = self.root_dir + "/logs/" + self.refs + ".log"
        self.logger      = loggerConfig(self.log_name, self.verbose)
        self.logger.warning("<===================================>")

        if self.visualize:
            self.vis = visdom.Visdom(port=self.visdom_port)
            #self.logger.warning("bash$: source activate pytorchenv")
            #self.logger.warning("bash$: python -m visdom.server")           # activate visdom server on bash
            #self.logger.warning("http://localhost:8097/env/" + self.refs)   # open this address on browser

class EnvParams(Params):    # settings for simulation environment
    def __init__(self):
        super(EnvParams, self).__init__()
        self.hei_state = 42
        self.wid_state = 42

class ModelParams(Params):  # settings for network architecture
    def __init__(self):
        super(ModelParams, self).__init__()

        self.state_shape = None # NOTE: set in fit_model of inherited Agents
        self.action_dim = None # NOTE: set in fit_model of inherited Agents

class AgentParams(Params):  # hyperparameters for drl agents
    def __init__(self):
        super(AgentParams, self).__init__()

        # optimizer
        self.optim          = SharedAdam    # share momentum across learners

        # hyperparameters
        self.steps               = 20000000 # max #iterations
        self.gamma               = 0.99
        self.clip_grad           = 40.
        self.lr                  = 0.001
        self.eval_freq           = 10       # NOTE: evaluation frequency in seconds
        self.eval_steps          = 2100    # Number of steps during testing. Each episode has a maximum of 2100 steps.
        self.prog_freq           = self.eval_freq
        self.test_nepisodes      = 10
        self.rollout_steps       = 20       # max look-ahead steps in a single rollout
        self.tau                 = 1.

        self.eta                 = 0.01     # scaling factor for intrinsic reward
        self.beta                = 0.2      # balance between inverse & forward model_type
        self.lmbda               = 0.1      # balance between A3C and ICM

        self.env_params    = EnvParams()
        self.model_params  = ModelParams()

class Options(Params):
    agent_params  = AgentParams()
