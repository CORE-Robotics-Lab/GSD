
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from gym.envs.mujoco import mujoco_env
from gym import utils
import numpy as np
import os


class HCCustom(HalfCheetahEnv):
    def __init__(self):
        if not hasattr(self, 'ctrl_weight'):
            self.ctrl_weight = 0.0
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -self.ctrl_weight * np.square(action).sum()
        current_vel = (xposafter - xposbefore) / self.dt
        # reward_run = current_vel
        reward_run = 1 if current_vel > 0 else -1
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, current_vel=current_vel)

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[1:],
                self.sim.data.qvel.flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

class HCCustomVel(HCCustom):
    def __init__(self, target_vel=None):
        self.target_vel = target_vel
        # static control weight
        self.ctrl_weight = 0.1
        # scaled control weight to allow for higher vels
        # 1 -> 5
        # 0.1 -> 0.04
        # self.ctrl_weight = ((5 - target_vel)/(5-1))*(0.2 - 0.04) + 0.04
        super().__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = -self.ctrl_weight * np.square(action).sum()
        current_vel = (xposafter - xposbefore) / self.dt
        # reward_run = current_vel
        reward_run = -np.square(current_vel/self.target_vel - 1) + 1
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl, current_vel=current_vel)
