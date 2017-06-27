from __future__ import absolute_import
from __future__ import division
import numpy as np
from gym.spaces.box import Box
import inspect
from collections import deque
from PIL import Image
import random
from vizdoom import *

from utils.helpers import Experience, rgb2y

class DoomEnv(object):
    def __init__(self, args, env_ind=0):
        self.logger     = args.logger
        self.ind        = env_ind               # NOTE: for creating multiple environment instances
        # general setup
        self.mode       = args.mode             # NOTE: save frames when mode=2
        self.seed       = args.seed + self.ind  # NOTE: so to give a different seed to each instance
        self.visualize  = args.visualize

        if self.visualize:
            self.vis        = args.vis
            self.refs       = args.refs
            self.win_state1 = "win_state1"

        self.neg_clip   = 0.0
        self.n          = 4
        self.skip       = 1

        self.hei_state = args.hei_state
        self.wid_state = args.wid_state
        assert self.hei_state == self.wid_state

        # most recent raw observations (for max pooling across time steps)
        self.obs_buffer = deque(maxlen=2)
        self.buffer = deque(maxlen=self.n)
        self.counter = 0  # init and reset should agree on this
        self.observation_space = Box(0.0, 255.0, (self.n,) + (self.hei_state,self.wid_state))
        self.scale = 1.0 / 255
        self.observation_space.high[...] = 1.0

        self.logger.warning("<-----------------------------------> Env")
        self.logger.warning("Creating Doom Game w/ Seed: " + str(self.seed))

        self._reset_experience()
        self.game = DoomGame()
        self.game.load_config("my_way_home.cfg")
        self.game.set_window_visible(False)
        self.game.set_seed(self.seed)
        self.game.init()

        # action space setup
        turn_left   = [1,0,0,0,0]
        turn_right  = [0,1,0,0,0]
        forward     = [0,0,1,0,0]
        move_left   = [0,0,0,1,0]
        move_right  = [0,0,0,0,1]
        self.actions = [turn_left, turn_right, forward, move_left, move_right]
        # state space setup

        self.logger.warning("State  Space: (" + str(self.state_shape) + " * " + str(self.state_shape) + ")")

    @property
    def state_shape(self):
        return self.hei_state

    @property
    def action_dim(self):
        return self.game.get_available_buttons_size()

    def visual(self):
        if self.visualize:
            self.win_state1 = self.vis.image(np.transpose(self.last_frame, (2, 0, 1)), env=self.refs, win=self.win_state1, opts=dict(title="state1"))
        if self.mode == 2:
            frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
            self.imsave(frame_name, self.last_frame)
            self.logger.warning("Saved  Frame    @ Step: " + str(self.frame_ind) + " To: " + frame_name)
            self.frame_ind += 1

    def sample_random_action(self):
        return random.choice(self.actions)

    def _observation(self, obs):
        obs = self._convert(obs)
        self.counter += 1
        if self.counter % self.skip == 0:
            self.buffer.append(obs)
        obsNew = np.stack(self.buffer, axis=0)
        return obsNew.astype(np.float32) * self.scale

    def _convert(self, obs):
        self.obs_buffer.append(obs)
        max_frame = np.max(np.stack(self.obs_buffer), axis=0)
        intensity_frame = rgb2y(max_frame).astype(np.uint8)
        small_frame = np.array(Image.fromarray(intensity_frame).resize((self.hei_state,self.wid_state), resample=Image.BILINEAR), dtype=np.uint8)
        return small_frame

    def reset(self):
        self._reset_experience()
        self.game.new_episode()
        """Clear buffer and re-fill by duplicating the first observation."""
        self.obs_buffer.clear()
        obs = self.game.get_state().screen_buffer
        obs_c = self._convert(obs)
        self.buffer.clear()
        self.counter = 0
        for _ in range(self.n - 1):
            self.buffer.append(np.zeros_like(obs_c))
        self.buffer.append(obs_c)
        obsNew = np.stack(self.buffer, axis=0)

        self.exp_state1 = obsNew.astype(np.float32) * self.scale
        return self._get_experience()

    def step(self, action_index):
        self.exp_action = action_index
        self.exp_reward = self.game.make_action(self.actions[self.exp_action])
        # Clip negative rewards to 0 as described in original paper
        self.exp_reward = self.neg_clip if self.exp_reward < self.neg_clip else self.exp_reward
        self.exp_terminal1 = self.game.is_episode_finished()
        if not self.exp_terminal1:
            self.exp_state1 = self._observation(self.game.get_state().screen_buffer)

        return self._get_experience()

    def _reset_experience(self):
        self.exp_state0     = None  # NOTE: always None in this module
        self.exp_action     = None
        self.exp_reward     = None
        self.exp_state1     = None
        self.exp_terminal1  = None

    def _get_experience(self):
        return Experience(state0 = self.exp_state0, # NOTE: here state0 is always None
                          action = self.exp_action,
                          reward = self.exp_reward,
                          state1 = self.exp_state1,
                          terminal1 = self.exp_terminal1)
