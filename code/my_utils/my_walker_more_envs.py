
from my_utils.my_walker_envs import WalkerCustom
import numpy as np

# range(1, 20)
class WalkerCustomForce(WalkerCustom):
    def __init__(self, param=None):
        self.step_count = -999
        self.force = np.zeros(9)
        self.force[4] = -10.0 * param
        self.timestep_start = 10
        self.timestep_end = 15
        WalkerCustom.__init__(self)
        self.step_count = None
    
    def reset(self, *args, **kwargs):
        self.step_count = -1
        return super().reset(*args, **kwargs)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        #######
        # only modified here
        self.step_count += 1
        if self.step_count >= self.timestep_start and self.step_count <= self.timestep_end:
            self.sim.data.qfrc_applied[:] = self.force
        #######
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

# range(10, 110, 10)
class WalkerCustomMotor(WalkerCustom):
    def __init__(self, param=None):
        self.step_count = -999
        self.timestep_start = 10
        self.timestep_end = self.timestep_start + param * 5
        WalkerCustom.__init__(self)
        self.step_count = None
    
    def reset(self, *args, **kwargs):
        self.step_count = -1
        return super().reset(*args, **kwargs)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        #######
        # only modified here
        self.step_count += 1
        if self.step_count >= self.timestep_start and self.step_count <= self.timestep_end:
            a[0] = 0
            a[1] = 0
        #######
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
