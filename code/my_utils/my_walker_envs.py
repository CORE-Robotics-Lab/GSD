
from gym.envs.mujoco.walker2d import Walker2dEnv
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import os


class WalkerCustom(Walker2dEnv):
    def __init__(self):
        # mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        mpath = os.path.join(os.path.dirname(__file__), self._get_local_model_path())
        mujoco_env.MujocoEnv.__init__(self, mpath, 4)
        utils.EzPickle.__init__(self)

    def _get_local_model_path(self):
        # copied from gym
        return "assets/walker2d.xml"

    def _is_valid(self, height, ang):
        # default from gym
        return (height > 0.8 and height < 2.0 and
                ang > -1.0 and ang < 1.0)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        reward = np.minimum((posafter - posbefore) / self.dt, 2)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (self._is_valid(height, ang))
        if done:
            reward = 0.0
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}

class WalkerCustomShortOne(WalkerCustom):
    def _get_local_model_path(self):
        return "assets/walker2d_short.xml"

class WalkerCustomShortTwo(WalkerCustom):
    def _get_local_model_path(self):
        return "assets/walker2d_short_orange.xml"

    def _is_valid(self, height, ang):
        return (height > 0.8 and height < 2.3 and
                ang > -1.2 and ang < 1.2)

class WalkerCustomShortLow(WalkerCustom):
    def _get_local_model_path(self):
        return "assets/walker2d_lowshort.xml"

class WalkerCustomShortHigh(WalkerCustom):
    def _get_local_model_path(self):
        return "assets/walker2d_shorthigh.xml"

    def _is_valid(self, height, ang):
        return (height > 0.8 and height < 2.3 and
                ang > -1.2 and ang < 1.2)

