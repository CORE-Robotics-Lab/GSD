import numpy as np
import gym
from gym import spaces
import random

class NormalizeGymWrapper(gym.ActionWrapper):
    """
    This wrapper normalize action to be in [-1, 1]
    """
    def __init__(self, env):
        super().__init__(env)

        high = self.env.action_space.high 
        low = self.env.action_space.low 
        
        np.testing.assert_array_equal(high, -low)   ## check that the original action bound is symmetric. 

        self.action_scale = np.abs(high) 
        
        self.action_space = spaces.Box(low=low / self.action_scale, high=high / self.action_scale)

    def step(self, action):

        action = action * self.action_scale # re-scale back to the original bound

        ob, reward, done, info = self.env.step(action)
        return ob, reward, done, info

class ClipGymWrapper(gym.ActionWrapper):
    """
    This wrapper clip action to be in [low, high].
    Cliped actions are required to prevent errors in box2d envs (LunarLander and BipedalWalker)
    """
    def __init__(self, env):
        super().__init__(env)
        
    def step(self, action):
        return self.env.step(np.clip(action, a_min=self.env.action_space.low, a_max=self.env.action_space.high))

class DetWrapper(gym.Wrapper):
    def __init__(self, env, seed=1234, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.reset_seed = seed
        self.step = self.env.step

    def reset(self, *args, **kwargs):
        self.env.seed(self.reset_seed)
        return self.env.reset(*args, **kwargs)

def get_preset_norm_params(env_name):
    if 'MazeCustom' == env_name:
        _mean = np.array([-0.6612632, -0.0017413688, 0.0, -0.6612632, -0.0017413688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        _var = np.array([0.19182448, 0.06888037, 0.0, 0.19182448, 0.06888037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        return _mean, _var
    if 'LongMazeCustom' == env_name:
        _mean = np.array([-0.013822945, 0.0074618785, 0.0, -0.013822945, 0.0074618785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        _var = np.array([1.3446114, 0.06647784, 0.0, 1.3446114, 0.06647784, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        return _mean, _var
    if 'SquareMazeCustom' == env_name:
        _mean = np.array([-0.10087447, -0.13619737, 0.0, -0.10087447, -0.13619737, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        _var = np.array([0.43043503, 0.48302662, 0.0, 0.43043503, 0.48302662, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        return _mean, _var
    if 'InvertedPendulumCustom' in env_name:
        _mean = np.array([0.0729858, -0.0016716026, 0.002245147, 1.5685651e-05, ])
        _var = np.array([0.129539, 0.0006043138, 0.03019733, 0.11344038, ])
        return _mean, _var
    if 'HopperCustom' in env_name:
        _mean = np.array([1.3272598, -0.010437929, -0.25985256, -0.20449784, 0.35017627, 1.0184759, 0.01053644, -0.0026972601, -0.035773784, -0.02577259, -0.08874242, ])
        _var = np.array([0.017411113, 0.0023247027, 0.015912948, 0.044308692, 0.17329663, 0.07066362, 1.8303837, 1.1087687, 2.674297, 2.6303196, 22.225044, ])
        return _mean, _var
    if 'WalkerCustom' in env_name:
        _mean = np.array([1.2105335, 0.20527387, -0.082111284, -0.6532193, -0.43052077, -0.40945193, -0.04873795, 0.44490942, 1.9410112, -0.0074631893, -0.1435331, -0.05915373, -1.3545277, -0.054890037, -0.27154154, -0.007139617, -0.9192989, ])
        _var = np.array([0.054656647, 1.0486019, 0.070256315, 0.69260675, 0.48571476, 0.5395569, 0.13277361, 0.40352994, 1.8134215, 1.1721958, 11.092487, 6.852541, 23.96774, 21.939192, 13.406528, 2.0018592, 21.718565, ])
        return _mean, _var
    if 'FetchPushDouble' in env_name:
        _mean = np.array([1.3393847, 0.7522395, 0.50469536, 1.372509, 0.75829595, 0.42492747, 1.3121672, 0.75229365, 0.42491522, -0.00019923452, 6.657027e-05, 7.728214e-05, 8.125672e-05, 8.7095614e-05, -9.0684443e-07, -3.874791e-06, 0.033142507, 0.0060635563, -0.07977553, -0.027228778, 4.3901695e-05, -0.079787955, 0.060370833, 0.006019797, 1.2451182e-05, -0.0001753179, 2.6303453e-05, 0.003922456, -5.3767042e-05, 5.086585e-05, -0.00017052342, ])
        _var = np.array([0.0006727087, 0.014243358, 0.0052448637, 1.4462184e-05, 0.035095036, 1.2150278e-07, 1.4052274e-05, 0.008615389, 1.053746e-07, 4.321604e-06, 7.6802506e-05, 8.884238e-05, 4.2492307e-08, 4.396905e-08, 3.450558e-08, 3.453689e-08, 0.000707859, 0.015792819, 0.0052576438, 0.0006994501, 0.009267948, 0.0052564824, 1.9297846e-05, 0.024111511, 2.3545135e-07, 7.124556e-05, 4.4745916e-06, 0.002859534, 4.2258907e-05, 5.590413e-06, 0.0039724302, ])
        return _mean, _var
    if 'FetchPushSingle' in env_name:
        _mean = np.array([1.338383, 0.8710547, 0.44276455, 1.3448265, 0.9941442, 0.2753455, 9.0386e-06, 0.0019883658, -0.0009659789, 9.475702e-05, 0.0064893994, -0.0037900023, 0.00017489163, 0.00021872863, 3.6580022e-05, -2.3745031e-05, 0.006437315, 0.12308938, -0.16741998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.41269404, 0.0022589213, -0.026833534, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        _var = np.array([3.6950576e-06, 0.03211417, 0.002868316, 0.00012070264, 0.072013706, 0.036038473, 2.907053e-08, 0.000149748, 5.746272e-05, 2.5594846e-07, 8.6957196e-05, 0.00026397538, 1.7862772e-07, 1.976884e-07, 1.08071994e-07, 1.13749095e-07, 0.00012457096, 0.017703079, 0.03201014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4418625, 9.4188865e-05, 0.012170918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ])
        return _mean, _var
    if 'FetchPickPlace' == env_name:
        _mean = np.array([1.3611525, 0.7493358, 0.48679742, 1.3506626, 0.7494836, 0.45666015, 0.00078684255, 8.490437e-06, -0.0009006164, 0.0019036992, 2.3394921e-05, -1.1395018e-05, 0.025977656, 0.025116548, 0.00048290702, 0.00046327902, -0.010488345, 0.00014697337, -0.030137885, ])
        _var = np.array([0.01150888, 0.011971201, 0.0058025713, 0.014026414, 0.012078155, 0.00385583, 6.518262e-05, 1.9300849e-05, 9.209539e-05, 3.885569e-05, 1.9853616e-05, 5.681303e-05, 0.00013171334, 0.00012701286, 5.3054766e-05, 5.3048756e-05, 0.0015327762, 7.4305103e-06, 0.0036456431, ])
        return _mean, _var
    if 'FetchPickPlaceWide' == env_name:
        _mean = np.array([1.3316312, 0.75053734, 0.48745787, 1.3216171, 0.75061256, 0.45728874, 0.00043296572, 2.9431474e-05, -0.00090324326, 0.0015572688, 4.8775484e-05, -1.13593605e-05, 0.025941554, 0.025112994, 0.0004818282, 0.00046077446, -0.010012019, 7.50878e-05, -0.030170027, 0.4050034, 0.47999793, 7.368857e-07, -5.4326025e-05, -0.0029905778, 0.059904125, 0.010838841, -0.66762686, -0.003350097, 1.2695923, 0.0032361625, 0.96887505, 0.0071729645, 0.025941554, 0.025112994, ])
        _var = np.array([0.0073414342, 0.040100683, 0.0058390615, 0.009173677, 0.040221445, 0.0039188806, 5.7635658e-05, 3.546051e-05, 9.199719e-05, 3.191135e-05, 3.662702e-05, 5.6675883e-05, 0.00013227799, 0.00012809616, 5.3041676e-05, 5.305033e-05, 0.0015434227, 7.6178017e-06, 0.0036438373, 1.2366996e-11, 4.2286175e-12, 6.889262e-14, 3.9720582e-10, 2.313568e-05, 1.6692263e-07, 0.07936911, 0.12651736, 8.829656e-05, 0.25340217, 9.093245e-05, 0.050546672, 0.07701173, 0.00013227799, 0.00012809616, ])
        return _mean, _var
    raise NotImplementedError(env_name)

def norm_arr(arr, mean, std, clip=10.0):
    arr_normed = (arr - mean) / std
    arr_clipped = np.clip(arr_normed, -clip, clip)
    return arr_clipped

def unnorm_arr(arr, mean, std):
    return arr * std + mean

class NormalizeObsWrapper(gym.ObservationWrapper):
    """
    This wrapper normalizes obs to be from N(0, 1) with clipping
    """
    def __init__(self, env, mean, var, clip=10.0, epsilon=1e-1):
        super().__init__(env)
        self.clip = clip
        # eps is not 1e-8, as dims where std ~ 0
        # upon changing cause a big jump in unnormed obs
        self.epsilon = epsilon
        self.norm_mean = mean
        self.norm_var = var
        # self.norm_std = np.sqrt(np.clip(var, np.square(self.epsilon), np.inf))
        self.norm_std = np.sqrt(var + self.epsilon)

    def observation(self, obs):
        return norm_arr(obs, self.norm_mean, self.norm_std, self.clip)

    def unnorm_obs(self, obs):
        return unnorm_arr(obs, self.norm_mean, self.norm_std)
