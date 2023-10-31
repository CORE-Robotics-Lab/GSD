
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
from gym.envs.mujoco.hopper import HopperEnv
from gym.envs.mujoco.humanoid import HumanoidEnv
from my_utils.mazes import MazeEnv
from gym.spaces import flatten_space, flatten

import numpy as np


class MazeCustom(MazeEnv):
    def __init__(self):
        super().__init__(n=1000)
        self.observation_space = flatten_space(self.observation_space['observation'])

    def reset(self, *args, **kwargs):
        return super().reset(*args, **kwargs)['observation']

    def step(self, action):
        obs, _, done, info = super().step(action)
        obs = obs['observation']
        rew = -np.linalg.norm(obs[:2] - np.array([-1.5, 0.0]))
        return obs, rew, done, info


class LongMazeCustom(MazeCustom):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._state['s0'] = np.array([-2, 0])
        self._state['state'] = np.array([-2, 0])
        self.trajectory = [self.state]
        return self._get_mdp_state()['observation']

    def step(self, action):
        obs, _, done, info = super().step(action)
        rew = self._state['state'][0] - self._state['prev_state'][0]
        return obs, rew, done, info

class SquareMazeCustom(MazeCustom):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._state['s0'] = np.array([-1, -0.8])
        self._state['state'] = np.array([-1, -0.8])
        self.trajectory = [self.state]
        return self._get_mdp_state()['observation']

    def step(self, action):
        obs, _, done, info = super().step(action)
        rew = 0
        return obs, rew, done, info

class CrossMazeCustom(MazeCustom):
    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        self._state['s0'] = np.array([-1, -1])
        self._state['state'] = np.array([-1, -1])
        self.trajectory = [self.state]
        return self._get_mdp_state()['observation']

    def step(self, action):
        obs, _, done, info = super().step(action)
        rew = 0
        return obs, rew, done, info

class InvertedPendulumCustom(InvertedPendulumEnv):
    def __init__(self, allow_early_termination=False):
        super().__init__()
        self.allow_early_termination = allow_early_termination

    def _pos_reward(self, ob):
        return 0

    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.3)
        reward += self._pos_reward(ob)
        done = not notdone
        if done:
            reward = 0.0
        done = False
        return ob, reward, done, dict(ramp_pos=ob[0])


class InvertedPendulumCustomPos(InvertedPendulumCustom):
    def __init__(self, target_pos=None):
        self.target_pos = target_pos
        self.weight_pos_reward = 0.6
        super().__init__()

    def _pos_reward(self, ob):
        return -np.abs(ob[0] - self.target_pos) * self.weight_pos_reward


class HopperCustom(HopperEnv):
    def __init__(self, allow_early_termination=False):
        super().__init__()
        self.allow_early_termination = allow_early_termination

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        # reward = (posafter - posbefore) / self.dt
        reward = np.minimum((posafter - posbefore) / self.dt, 1)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (
            np.isfinite(s).all()
            and (np.abs(s[2:]) < 100).all()
            and (height > 0.7)
            and (abs(ang) < 0.2)
        )
        if done:
            reward = 0.0
        done = False
        ob = self._get_obs()
        return ob, reward, done, {}


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidCustom(HumanoidEnv):
    def step(self, a):
        pos_before = mass_center(self.model, self.sim)
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = np.minimum(0.25 * (pos_after - pos_before) / self.model.opt.timestep, 1)
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.sim.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        if done:
            reward = 0
        done = False
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)
