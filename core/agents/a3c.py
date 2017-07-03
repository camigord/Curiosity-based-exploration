from __future__ import absolute_import
from __future__ import division
import torch.multiprocessing as mp

import torch
import torch.optim as optim

from utils.helpers import Experience
from core.agents.a3cSingleProcess import A3CLearner, A3CEvaluator, A3CTester

class A3CAgent(object):
    def __init__(self, args, env_prototype, model_prototype):
        super(A3CAgent, self).__init__()
        self.logger = args.logger

        self.logger.warning("<===================================> A3C-Master {Env(dummy) & Model}")

        self.args = args
        self.env_prototype = env_prototype
        self.model_prototype = model_prototype
        self.num_processes = args.num_processes
        self.optim = args.optim

        if self.args.visualize:
            self.vis = args.vis
            self.refs = args.refs

        # dummy_env just to get state_shape & action_dim
        self.dummy_env   = self.env_prototype(self.args.env_params, self.args.num_processes)
        self.state_shape = self.dummy_env.state_shape
        self.action_dim  = self.dummy_env.action_dim
        del self.dummy_env

        # global shared model
        self.model_params = args.model_params
        self.model_params.state_shape = self.state_shape
        self.model_params.action_dim  = self.action_dim
        self.model = self.model_prototype(self.model_params)
        self._load_model(self.args.model_file)   # load pretrained model if provided
        self.model.share_memory()

        self.optimizer = self.args.optim(self.model.parameters(), lr = self.args.lr)
        self.optimizer.share_memory()

        # global counters
        self.frame_step   = mp.Value('l', 0) # global frame step counter
        self.train_step   = mp.Value('l', 0) # global train step counter
        # global training stats
        self.p_loss_avg         = mp.Value('d', 0.) # global policy loss
        self.v_loss_avg         = mp.Value('d', 0.) # global value loss
        self.inverse_loss_avg   = mp.Value('d', 0.) # global icm_loss
        self.forward_loss_avg   = mp.Value('d', 0.) # global icm_loss
        self.icm_loss_avg       = mp.Value('d', 0.) # global icm_loss
        self.loss_avg           = mp.Value('d', 0.) # global loss
        self.loss_counter       = mp.Value('l', 0)  # storing this many losses
        self._reset_training_loggings()

    def _reset_experience(self):
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False)

    def _load_model(self, model_file):
        if model_file:
            self.logger.warning("Loading Model: " + self.args.model_file + " ...")
            self.model.load_state_dict(torch.load(model_file))
            self.logger.warning("Loaded  Model: " + self.args.model_file + " ...")
        else:
            self.logger.warning("No Pretrained Model. Will Train From Scratch.")

    def _save_model(self, step):
        self.logger.warning("Saving Model    @ Step: " + str(step) + ": " + self.args.model_name + " ...")
        torch.save(self.model.state_dict(), self.args.model_name)
        self.logger.warning("Saved  Model    @ Step: " + str(step) + ": " + self.args.model_name + ".")

    def _reset_training_loggings(self):
        self.p_loss_avg.value           = 0.
        self.v_loss_avg.value           = 0.
        self.loss_avg.value             = 0.
        self.loss_counter.value         = 0
        self.inverse_loss_avg.value     = 0
        self.forward_loss_avg.value     = 0
        self.icm_loss_avg.value         = 0

    def fit_model(self):
        self.jobs = []
        for process_id in range(self.num_processes):
            self.jobs.append(A3CLearner(self, process_id))
        self.jobs.append(A3CEvaluator(self, self.num_processes))

        self.logger.warning("<===================================> Training ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def test_model(self):
        self.jobs = []
        self.jobs.append(A3CTester(self))

        self.logger.warning("<===================================> Testing ...")
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()
