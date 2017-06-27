from __future__ import absolute_import
from __future__ import division
import numpy as np
import random
import time
import math
import torch
import torch.multiprocessing as mp
from torch.autograd import Variable
import torch.nn.functional as F

from utils.helpers import Experience, AugmentedExperience

class A3CSingleProcess(mp.Process):
    def __init__(self, master, process_id=0):
        super(A3CSingleProcess, self).__init__(name = "Process-%d" % process_id)
        # NOTE: self.master.* refers to parameters shared across all processes
        # NOTE: self.*        refers to process-specific properties
        # NOTE: we are not copying self.master.* to self.* to keep the code clean

        self.master = master
        self.process_id = process_id

        # env
        self.env = self.master.env_prototype(self.master.args.env_params, self.process_id)
        # model
        self.model = self.master.model_prototype(self.master.args.model_params)
        self._sync_local_with_global()

        # experience
        self._reset_experience()

        # lstm hidden states
        self._reset_lstm_hidden_vb_episode() # clear up hidden state
        self._reset_lstm_hidden_vb_rollout() # detach the previous variable from the computation graph

        self.master.logger.warning("Registered A3C-SingleProcess-Agent #" + str(self.process_id) + " w/ Env (seed:" + str(self.env.seed) + ").")

    def _reset_experience(self):    # for getting one set of observation from env for every action taken
        self.experience = Experience(state0 = None,
                                     action = None,
                                     reward = None,
                                     state1 = None,
                                     terminal1 = False) # TODO: should check this again

    # NOTE: to be called at the beginning of each new episode, clear up the hidden state
    def _reset_lstm_hidden_vb_episode(self, training=True):
        not_training = not training
        self.lstm_hidden_vb = (Variable(torch.zeros(1, self.master.args.hidden_dim).type(self.master.args.dtype), volatile=not_training),
                               Variable(torch.zeros(1, self.master.args.hidden_dim).type(self.master.args.dtype), volatile=not_training))

    # NOTE: to be called at the beginning of each rollout, detach the previous variable from the graph
    def _reset_lstm_hidden_vb_rollout(self):
        self.lstm_hidden_vb = (Variable(self.lstm_hidden_vb[0].data),
                               Variable(self.lstm_hidden_vb[1].data))

    def _sync_local_with_global(self):  # grab the current global model for local learning/evaluating
        self.model.load_state_dict(self.master.model.state_dict())

    def _preprocessState(self, state, is_volatile=False):
        state_vb = Variable(torch.from_numpy(state).unsqueeze(0).type(self.master.args.dtype), volatile=is_volatile)
        return state_vb

    def _forward(self, state_vb):
        p_vb, v_vb, self.lstm_hidden_vb = self.model((state_vb, self.lstm_hidden_vb), icm = False)

        if self.training:
            action = p_vb.multinomial().data[0][0]
        else:
            action = p_vb.max(1)[1].data.squeeze().numpy()[0]

        return action, p_vb, v_vb

    def run(self):
        raise NotImplementedError("not implemented in base class")

class A3CLearner(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> A3C-Learner #" + str(process_id) + " {Env & Model}")
        super(A3CLearner, self).__init__(master, process_id)

        self.optimizer = self.master.optim(self.model.parameters(), lr = self.master.args.lr)

        self._reset_rollout()

        self.training = True    # choose actions by polinomial
        # local counters
        self.frame_step   = 0   # local frame step counter
        self.train_step   = 0   # local train step counter
        # local training stats
        self.p_loss_avg   = 0.  # global policy loss
        self.v_loss_avg   = 0.  # global value loss
        self.loss_avg     = 0.  # global value loss
        self.forward_loss = 0
        self.inverse_loss = 0
        self.ICM_loss     = 0   # ICM module loss (inverse and forward models)
        self.loss_counter = 0   # storing this many losses
        self._reset_training_loggings()

        # copy local training stats to global every prog_freq
        self.last_prog = time.time()

    def _reset_training_loggings(self):
        self.p_loss_avg   = 0.
        self.v_loss_avg   = 0.
        self.loss_avg     = 0.
        self.loss_counter = 0
        self.ICM_loss     = 0
        self.forward_loss = 0
        self.inverse_loss = 0

    def _reset_rollout(self):       # for storing the experiences collected through one rollout
        self.rollout = AugmentedExperience(state0 = [],
                                           action = [],
                                           one_hot_action = [],
                                           total_reward = [],
                                           inverse_out = [],
                                           forward_out = [],
                                           vec_state1 = [],
                                           state1 = [],
                                           terminal1 = [],
                                           policy_vb = [],
                                           sigmoid_vb = [],
                                           value0_vb = [])

    # NOTE: since no backward passes has ever been run on the global model
    # NOTE: its grad has never been initialized, here we ensure proper initialization
    # NOTE: reference: https://discuss.pytorch.org/t/problem-on-variable-grad-data/957
    def _ensure_global_grads(self):
        for global_param, local_param in zip(self.master.model.parameters(),self.model.parameters()):
            if global_param.grad is not None:
                return
            global_param._grad = local_param.grad

    def _get_valueT_vb(self):
        if self.rollout.terminal1[-1]:  # for terminal sT
            valueT_vb = Variable(torch.zeros(1, 1))
        else:                           # for non-terminal sT
            sT_vb = self._preprocessState(self.rollout.state1[-1], True)        # bootstrap from last state

            _, valueT_vb, _ = self.model((sT_vb, self.lstm_hidden_vb), icm=False)    # NOTE: only doing inference here
            valueT_vb = Variable(valueT_vb.data)

        return valueT_vb

    def _backward(self):
        # preparation
        rollout_steps = len(self.rollout.total_reward)

        if self.process_id ==0:
            print("Entering backward:", rollout_steps)

        policy_vb = self.rollout.policy_vb

        action_batch_vb = Variable(torch.from_numpy(np.array(self.rollout.action)).long())

        policy_log_vb = [torch.log(policy_vb[i]) for i in range(rollout_steps)]
        entropy_vb    = [- (policy_log_vb[i] * policy_vb[i]).sum(1) for i in range(rollout_steps)]
        policy_log_vb = [policy_log_vb[i].gather(1, action_batch_vb[i].unsqueeze(0)) for i in range(rollout_steps) ]

        valueT_vb     = self._get_valueT_vb()
        self.rollout.value0_vb.append(Variable(valueT_vb.data)) # NOTE: only this last entry is Volatile, all others are still in the graph
        gae_ts        = torch.zeros(1, 1)

        # compute loss
        policy_loss_vb = Variable(torch.zeros(1, 1))
        value_loss_vb  = Variable(torch.zeros(1, 1))
        inverse_loss_vb = Variable(torch.zeros(1, 1))
        forward_loss_vb = Variable(torch.zeros(1, 1))
        for i in reversed(range(rollout_steps)):
            # 1) Policy loss
            valueT_vb     = self.master.args.gamma * valueT_vb + self.rollout.total_reward[i]
            advantage_vb  = valueT_vb - self.rollout.value0_vb[i]
            value_loss_vb = value_loss_vb + 0.5 * advantage_vb.pow(2)

            # Generalized Advantage Estimation
            tderr_ts = self.rollout.total_reward[i] + self.master.args.gamma * self.rollout.value0_vb[i + 1].data - self.rollout.value0_vb[i].data
            gae_ts   = self.master.args.gamma * gae_ts * self.master.args.tau + tderr_ts

            policy_loss_vb = policy_loss_vb - policy_log_vb[i] * Variable(gae_ts) - 0.01 * entropy_vb[i]

            # 2) Loss for inverse model
            cross_entropy = - (self.rollout.one_hot_action[i] * torch.log(self.rollout.inverse_out[i] + 1e-15)).sum(1)
            inverse_loss_vb = inverse_loss_vb + cross_entropy

            # 3) Loss for forward model
            forward_err = self.rollout.forward_out[i] - self.rollout.vec_state1[i]
            forward_loss_vb = forward_loss_vb + 0.5 * (forward_err.pow(2)).sum(1)

        ICM_loss_vb = (1-self.master.args.beta) * inverse_loss_vb + self.master.args.beta * forward_loss_vb
        ICM_loss_vb.backward(retain_variables=True)
        loss_vb = self.master.args.lmbda * (policy_loss_vb + 0.5 * value_loss_vb)
        loss_vb.backward()

        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.master.args.clip_grad)

        self._ensure_global_grads()
        self.master.optimizer.step()
        self.train_step += 1
        self.master.train_step.value += 1

        # log training stats
        self.p_loss_avg   += policy_loss_vb.data.numpy()
        self.v_loss_avg   += value_loss_vb.data.numpy()
        self.loss_avg     += loss_vb.data.numpy()
        self.ICM_loss     += ICM_loss_vb.data.numpy()
        self.forward_loss += forward_loss_vb.data.numpy()
        self.inverse_loss += inverse_loss_vb.data.numpy()
        self.loss_counter += 1

    def _rollout(self, episode_steps, episode_reward):
        t_start = self.frame_step
        # continue to rollout only if:
        # 1. not running out of max steps of this current rollout, and
        # 2. not terminal, and
        # 3. not exceeding max steps of this current episode
        # 4. master not exceeding max train steps
        while (self.frame_step - t_start) < self.master.args.rollout_steps and not self.experience.terminal1:
            # NOTE: here first store the last frame: experience.state1 as rollout.state0

            state_0 = self.experience.state1
            self.rollout.state0.append(state_0)

            # then get the action to take from rollout.state0 (experience.state1)
            action, p_vb, v_vb = self._forward(self._preprocessState(self.experience.state1))

            # then execute action in env to get a new experience.state1 -> rollout.state1
            self.experience = self.env.step(action)

            # One-hot action for ICM module
            oh_action = torch.Tensor(1, self.master.action_dim)
            oh_action.zero_()
            oh_action.scatter_(1,torch.LongTensor([[action]]),1)
            oh_action = Variable(oh_action)

            # Forward pass through ICM module
            vec_st1, inverse, forward = self.model((self._preprocessState(state_0),self._preprocessState(self.experience.state1),oh_action),icm = True)

            # Compute intrinsic reward
            reward_intrinsic = self.master.args.eta * ((vec_st1 - forward).pow(2)).sum(1) / 2.
            reward_intrinsic = reward_intrinsic.data.numpy()[0][0]

            # push experience into rollout
            self.rollout.action.append(action)
            self.rollout.one_hot_action.append(oh_action)
            self.rollout.total_reward.append(self.experience.reward + reward_intrinsic)
            self.rollout.state1.append(self.experience.state1)
            self.rollout.terminal1.append(self.experience.terminal1)
            self.rollout.policy_vb.append(p_vb)
            self.rollout.value0_vb.append(v_vb)
            self.rollout.inverse_out.append(inverse)
            self.rollout.forward_out.append(forward)
            self.rollout.vec_state1.append(vec_st1)

            episode_steps += 1
            episode_reward += self.experience.reward
            self.frame_step += 1
            self.master.frame_step.value += 1

            # NOTE: we put this condition inside to make sure this current rollout won't be empty
            if self.master.train_step.value >= self.master.args.steps:
                break

        return episode_steps, episode_reward

    def run(self):
        # make sure processes are not completely synced by sleeping a bit
        time.sleep(int(np.random.rand() * (self.process_id + 5)))

        nepisodes = 0
        nepisodes_solved = 0
        episode_steps = None
        episode_reward = None
        should_start_new = True
        while self.master.train_step.value < self.master.args.steps:
            # sync in every step
            self._sync_local_with_global()
            self.optimizer.zero_grad()
            # reset rollout experiences
            self._reset_rollout()

            # start of a new episode
            if should_start_new:
                episode_steps = 0
                episode_reward = 0.
                # NOTE: clear hidden state at the beginning of each episode
                self._reset_lstm_hidden_vb_episode()

                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                # reset flag
                should_start_new = False

            # NOTE: detach the previous hidden variable from the graph at the beginning of each rollout
            self._reset_lstm_hidden_vb_rollout()

            # Run a rollout for rollout_steps or until terminal
            episode_steps, episode_reward = self._rollout(episode_steps, episode_reward)

            if self.experience.terminal1:
                if self.process_id ==0:
                    print('//////////////////////////////')
                    print("Finish episode, starting:", nepisodes)

                nepisodes += 1
                should_start_new = True
                if self.experience.terminal1:
                    nepisodes_solved += 1

            # calculate loss
            self._backward()
            if self.process_id ==0:
                print("Finish backbard pass:", episode_steps)

            # copy local training stats to global at prog_freq, and clear up local stats
            if time.time() - self.last_prog >= self.master.args.prog_freq:
                self.master.p_loss_avg.value   += self.p_loss_avg
                self.master.v_loss_avg.value   += self.v_loss_avg
                self.master.loss_avg.value     += self.loss_avg
                self.master.inverse_loss_avg.value   += self.inverse_loss
                self.master.forward_loss_avg.value   += self.forward_loss
                self.master.icm_loss_avg.value       += self.ICM_loss
                self.master.loss_counter.value += self.loss_counter
                self._reset_training_loggings()
                self.last_prog = time.time()

        self.master.logger.warning("////////////////////////////////////////")
        self.master.logger.warning("TERMINATING!", self.process_id)

class A3CEvaluator(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> A3C-Evaluator {Env & Model}")
        super(A3CEvaluator, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self._reset_loggings()

        self.start_time = time.time()
        self.last_eval = time.time()

    def _reset_loggings(self):
        # training stats across all processes
        self.p_loss_avg_log = []
        self.v_loss_avg_log = []
        self.loss_avg_log = []
        self.inverse_loss_avg_log = []
        self.forward_loss_avg_log = []
        self.icm_loss_avg_log = []
        # evaluation stats
        self.entropy_avg_log = []
        self.v_avg_log = []
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.master.args.visualize:
            # training stats across all processes
            self.win_p_loss_avg = "win_p_loss_avg"
            self.win_v_loss_avg = "win_v_loss_avg"
            self.win_loss_avg = "win_loss_avg"
            self.win_forward_loss_avg = "win_forward_loss_avg"
            self.win_inverse_loss_avg = "win_inverse_loss_avg"
            self.win_icm_loss_avg = "win_icm_loss_avg"
            # evaluation stats
            self.win_entropy_avg = "win_entropy_avg"
            self.win_v_avg = "win_v_avg"
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def _eval_model(self):
        self.last_eval = time.time()
        eval_at_train_step = self.master.train_step.value
        eval_at_frame_step = self.master.frame_step.value

        # first grab the latest global model to do the evaluation
        self._sync_local_with_global()

        # evaluate
        eval_step = 0

        eval_entropy_log = []
        eval_v_log = []
        eval_nepisodes = 0
        eval_nepisodes_solved = 0
        eval_episode_steps = None
        eval_episode_steps_log = []
        eval_episode_reward = None
        eval_episode_reward_log = []
        eval_should_start_new = True

        while eval_step < self.master.args.eval_steps:
            if eval_should_start_new:   # start of a new episode
                eval_episode_steps = 0
                eval_episode_reward = 0.
                # reset lstm_hidden_vb for new episode
                # NOTE: clear hidden state at the beginning of each episode
                self._reset_lstm_hidden_vb_episode(self.training)

                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None

                #if self.master.args.visualize: self.env.visual()
                #if self.master.args.render: self.env.render()
                # reset flag
                eval_should_start_new = False

            # NOTE: detach the previous hidden variable from the graph at the beginning of each step
            # NOTE: not necessary here in evaluation but we do it anyways
            self._reset_lstm_hidden_vb_rollout()

            # Run a single step
            eval_action, p_vb, v_vb = self._forward(self._preprocessState(self.experience.state1))

            self.experience = self.env.step(eval_action)

            #if self.master.args.visualize: self.env.visual()
            #if self.master.args.render: self.env.render()

            if self.experience.terminal1 or (eval_step + 1) == self.master.args.eval_steps:
                eval_should_start_new = True

            eval_episode_steps += 1
            eval_episode_reward += self.experience.reward
            eval_step += 1

            if eval_should_start_new:
                eval_nepisodes += 1
                if self.experience.terminal1:
                    eval_nepisodes_solved += 1

                # This episode is finished, report and reset
                eval_entropy_log.append([np.mean((-torch.log(p_vb.data.squeeze()) * p_vb.data.squeeze()).numpy())])

                eval_v_log.append([v_vb.data.numpy()])
                eval_episode_steps_log.append([eval_episode_steps])
                eval_episode_reward_log.append([eval_episode_reward])
                self._reset_experience()
                eval_episode_steps = None
                eval_episode_reward = None

        # Logging for this evaluation phase
        loss_counter = self.master.loss_counter.value
        p_loss_avg = self.master.p_loss_avg.value / loss_counter if loss_counter > 0 else -30.
        v_loss_avg = self.master.v_loss_avg.value / loss_counter if loss_counter > 0 else -30.
        loss_avg = self.master.loss_avg.value / loss_counter if loss_counter > 0 else -30.
        inverse_loss_avg = self.master.inverse_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        forward_loss_avg = self.master.forward_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        icm_loss_avg = self.master.icm_loss_avg.value / loss_counter if loss_counter > 0 else 0.
        self.master._reset_training_loggings()

        self.p_loss_avg_log.append([eval_at_train_step, p_loss_avg])
        self.v_loss_avg_log.append([eval_at_train_step, v_loss_avg])
        self.loss_avg_log.append([eval_at_train_step, loss_avg])

        self.inverse_loss_avg_log.append([eval_at_train_step, inverse_loss_avg])
        self.forward_loss_avg_log.append([eval_at_train_step, forward_loss_avg])
        self.icm_loss_avg_log.append([eval_at_train_step, icm_loss_avg])


        self.entropy_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_entropy_log))])
        self.v_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_v_log))])
        self.steps_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_episode_steps_log))])
        self.steps_std_log.append([eval_at_train_step, np.std(np.asarray(eval_episode_steps_log))]); del eval_episode_steps_log
        self.reward_avg_log.append([eval_at_train_step, np.mean(np.asarray(eval_episode_reward_log))])
        self.reward_std_log.append([eval_at_train_step, np.std(np.asarray(eval_episode_reward_log))]); del eval_episode_reward_log
        self.nepisodes_log.append([eval_at_train_step, eval_nepisodes])
        self.nepisodes_solved_log.append([eval_at_train_step, eval_nepisodes_solved])
        self.repisodes_solved_log.append([eval_at_train_step, (eval_nepisodes_solved/eval_nepisodes) if eval_nepisodes > 0 else 0.])

        # plotting
        if self.master.args.visualize:
            self.win_p_loss_avg = self.master.vis.scatter(X=np.array(self.p_loss_avg_log), env=self.master.refs, win=self.win_p_loss_avg, opts=dict(title="p_loss_avg"))
            self.win_v_loss_avg = self.master.vis.scatter(X=np.array(self.v_loss_avg_log), env=self.master.refs, win=self.win_v_loss_avg, opts=dict(title="v_loss_avg"))
            self.win_loss_avg = self.master.vis.scatter(X=np.array(self.loss_avg_log), env=self.master.refs, win=self.win_loss_avg, opts=dict(title="loss_avg"))

            self.win_forward_loss_avg = self.master.vis.scatter(X=np.array(self.forward_loss_avg_log), env=self.master.refs, win=self.win_forward_loss_avg, opts=dict(title="forward_loss_avg"))
            self.win_inverse_loss_avg = self.master.vis.scatter(X=np.array(self.inverse_loss_avg_log), env=self.master.refs, win=self.win_inverse_loss_avg, opts=dict(title="inverse_loss_avg"))
            self.win_icm_loss_avg = self.master.vis.scatter(X=np.array(self.icm_loss_avg_log), env=self.master.refs, win=self.win_icm_loss_avg, opts=dict(title="icm_loss_avg"))


            self.win_entropy_avg = self.master.vis.scatter(X=np.array(self.entropy_avg_log), env=self.master.refs, win=self.win_entropy_avg, opts=dict(title="entropy_avg"))
            self.win_v_avg = self.master.vis.scatter(X=np.array(self.v_avg_log), env=self.master.refs, win=self.win_v_avg, opts=dict(title="v_avg"))
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.master.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.master.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))

        # logging
        self.master.logger.warning("Reporting       @ Step: " + str(eval_at_train_step) + " | Elapsed Time: " + str(time.time() - self.start_time))
        self.master.logger.warning("Iteration: {}; p_loss_avg: {}".format(eval_at_train_step, self.p_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; v_loss_avg: {}".format(eval_at_train_step, self.v_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; loss_avg: {}".format(eval_at_train_step, self.loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; forward_loss_avg: {}".format(eval_at_train_step, self.forward_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; inverse_loss_avg: {}".format(eval_at_train_step, self.inverse_loss_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; icm_loss_avg: {}".format(eval_at_train_step, self.icm_loss_avg_log[-1][1]))
        self.master._reset_training_loggings()

        self.master.logger.warning("Evaluating      @ Step: " + str(eval_at_train_step) + " | (" + str(eval_at_frame_step) + " frames)...")
        self.master.logger.warning("Evaluation        Took: " + str(time.time() - self.last_eval))
        self.master.logger.warning("Iteration: {}; entropy_avg: {}".format(eval_at_train_step, self.entropy_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; v_avg: {}".format(eval_at_train_step, self.v_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; steps_avg: {}".format(eval_at_train_step, self.steps_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; steps_std: {}".format(eval_at_train_step, self.steps_std_log[-1][1]))
        self.master.logger.warning("Iteration: {}; reward_avg: {}".format(eval_at_train_step, self.reward_avg_log[-1][1]))
        self.master.logger.warning("Iteration: {}; reward_std: {}".format(eval_at_train_step, self.reward_std_log[-1][1]))
        self.master.logger.warning("Iteration: {}; nepisodes: {}".format(eval_at_train_step, self.nepisodes_log[-1][1]))
        self.master.logger.warning("Iteration: {}; nepisodes_solved: {}".format(eval_at_train_step, self.nepisodes_solved_log[-1][1]))
        self.master.logger.warning("Iteration: {}; repisodes_solved: {}".format(eval_at_train_step, self.repisodes_solved_log[-1][1]))

        self.last_eval = time.time()

        # save model
        self.master._save_model(eval_at_train_step)

    def run(self):
        while self.master.train_step.value < self.master.args.steps:
            if time.time() - self.last_eval > self.master.args.eval_freq:
                self._eval_model()
        # we also do a final evaluation after training is done
        self._eval_model()

class A3CTester(A3CSingleProcess):
    def __init__(self, master, process_id=0):
        master.logger.warning("<===================================> A3C-Tester {Env & Model}")
        super(A3CTester, self).__init__(master, process_id)

        self.training = False   # choose actions w/ max probability
        self._reset_loggings()

        self.start_time = time.time()

    def _reset_loggings(self):
        # testing stats
        self.steps_avg_log = []
        self.steps_std_log = []
        self.reward_avg_log = []
        self.reward_std_log = []
        self.nepisodes_log = []
        self.nepisodes_solved_log = []
        self.repisodes_solved_log = []
        # placeholders for windows for online curve plotting
        if self.master.args.visualize:
            # evaluation stats
            self.win_steps_avg = "win_steps_avg"
            self.win_steps_std = "win_steps_std"
            self.win_reward_avg = "win_reward_avg"
            self.win_reward_std = "win_reward_std"
            self.win_nepisodes = "win_nepisodes"
            self.win_nepisodes_solved = "win_nepisodes_solved"
            self.win_repisodes_solved = "win_repisodes_solved"

    def run(self):
        test_step = 0
        test_nepisodes = 0
        test_nepisodes_solved = 0
        test_episode_steps = None
        test_episode_steps_log = []
        test_episode_reward = None
        test_episode_reward_log = []
        test_should_start_new = True
        while test_nepisodes < self.master.test_nepisodes:
            if test_should_start_new:   # start of a new episode
                test_episode_steps = 0
                test_episode_reward = 0.
                # reset lstm_hidden_vb for new episode
                if self.master.enable_lstm:
                    # NOTE: clear hidden state at the beginning of each episode
                    self._reset_lstm_hidden_vb_episode(self.training)
                # Obtain the initial observation by resetting the environment
                self._reset_experience()
                self.experience = self.env.reset()
                assert self.experience.state1 is not None
                if not self.training:
                    if self.master.args.visualize: self.env.visual()
                    if self.master.render: self.env.render()
                # reset flag
                test_should_start_new = False
            if self.master.enable_lstm:
                # NOTE: detach the previous hidden variable from the graph at the beginning of each step
                # NOTE: not necessary here in testing but we do it anyways
                self._reset_lstm_hidden_vb_rollout()
            # Run a single step
            test_action, p_vb, v_vb = self._forward(self._preprocessState(self.experience.state1, True))

            self.experience = self.env.step(test_action)
            if not self.training:
                if self.master.args.visualize: self.env.visual()
                if self.master.render: self.env.render()
            if self.experience.terminal1:
                test_should_start_new = True

            test_episode_steps += 1
            test_episode_reward += self.experience.reward
            test_step += 1

            if test_should_start_new:
                test_nepisodes += 1
                if self.experience.terminal1:
                    test_nepisodes_solved += 1

                # This episode is finished, report and reset
                test_episode_steps_log.append([test_episode_steps])
                test_episode_reward_log.append([test_episode_reward])
                self._reset_experience()
                test_episode_steps = None
                test_episode_reward = None

        self.steps_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_steps_log))])
        self.steps_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_steps_log))]); del test_episode_steps_log
        self.reward_avg_log.append([test_nepisodes, np.mean(np.asarray(test_episode_reward_log))])
        self.reward_std_log.append([test_nepisodes, np.std(np.asarray(test_episode_reward_log))]); del test_episode_reward_log
        self.nepisodes_log.append([test_nepisodes, test_nepisodes])
        self.nepisodes_solved_log.append([test_nepisodes, test_nepisodes_solved])
        self.repisodes_solved_log.append([test_nepisodes, (test_nepisodes_solved/test_nepisodes) if test_nepisodes > 0 else 0.])
        # plotting
        if self.master.args.visualize:
            self.win_steps_avg = self.master.vis.scatter(X=np.array(self.steps_avg_log), env=self.master.refs, win=self.win_steps_avg, opts=dict(title="steps_avg"))
            # self.win_steps_std = self.master.vis.scatter(X=np.array(self.steps_std_log), env=self.master.refs, win=self.win_steps_std, opts=dict(title="steps_std"))
            self.win_reward_avg = self.master.vis.scatter(X=np.array(self.reward_avg_log), env=self.master.refs, win=self.win_reward_avg, opts=dict(title="reward_avg"))
            # self.win_reward_std = self.master.vis.scatter(X=np.array(self.reward_std_log), env=self.master.refs, win=self.win_reward_std, opts=dict(title="reward_std"))
            self.win_nepisodes = self.master.vis.scatter(X=np.array(self.nepisodes_log), env=self.master.refs, win=self.win_nepisodes, opts=dict(title="nepisodes"))
            self.win_nepisodes_solved = self.master.vis.scatter(X=np.array(self.nepisodes_solved_log), env=self.master.refs, win=self.win_nepisodes_solved, opts=dict(title="nepisodes_solved"))
            self.win_repisodes_solved = self.master.vis.scatter(X=np.array(self.repisodes_solved_log), env=self.master.refs, win=self.win_repisodes_solved, opts=dict(title="repisodes_solved"))
        # logging
        self.master.logger.warning("Testing  Took: " + str(time.time() - self.start_time))
        self.master.logger.warning("Testing: steps_avg: {}".format(self.steps_avg_log[-1][1]))
        self.master.logger.warning("Testing: steps_std: {}".format(self.steps_std_log[-1][1]))
        self.master.logger.warning("Testing: reward_avg: {}".format(self.reward_avg_log[-1][1]))
        self.master.logger.warning("Testing: reward_std: {}".format(self.reward_std_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes: {}".format(self.nepisodes_log[-1][1]))
        self.master.logger.warning("Testing: nepisodes_solved: {}".format(self.nepisodes_solved_log[-1][1]))
        self.master.logger.warning("Testing: repisodes_solved: {}".format(self.repisodes_solved_log[-1][1]))
