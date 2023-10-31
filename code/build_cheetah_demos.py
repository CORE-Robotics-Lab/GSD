
from vild_main import make_env
from args_parser import * 

import gym
import torch
import h5py
import os
import numpy as np
import time
np.set_printoptions(suppress=True)

args = args_parser()
env_name = args.env_name
args.norm_obs = 0
env = make_env(args)[0]

RENDER = args.render
CHEETAH = True

if CHEETAH:
    latent_values = np.arange(0, 5)+1
    vild_modelpath_template = 'logsarchive/hcv1/HCCustomVel{}0/ckpt_policy_T1000000.pt'
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])
    n_episodes = 5

rollouts = []
for latent in latent_values:
    if CHEETAH:
        # vild_modelpath = vild_modelpath_template.format(3)
        vild_modelpath = vild_modelpath_template.format(int(latent))
        args.rl_method = 'sac'
        args.hidden_size = [100, 100]
        args.activation = 'relu'

        from core.ac import SAC
        # load model from checkpoint
        vild_agent = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=action_bound, encode_dim=0)
        vild_agent.load_model(vild_modelpath)
        vild_agent.policy_to_device(device_cpu) 
        print(latent, "Policy model is loaded from %s" % vild_modelpath)

    for i in range(n_episodes):
        obs, done = env.reset(), False
        obss_list = [obs.copy()]
        acts_list = []
        rews_list = []
        info_list = []

        t = 0
        while not done:

            if CHEETAH:
                with torch.no_grad():
                    obs_th = torch.Tensor(obs)
                    action_th = vild_agent.greedy_action(obs_th)
                    action = action_th.data.numpy()

            # action += np.random.randn(*action.shape)*0.01
            obs, rew, done, info = env.step(action)

            obss_list.append(obs.copy())
            acts_list.append(action.copy())
            rews_list.append(rew)
            info_list.append(info)

            # print(t, obs[:2], error[:2])

            if RENDER:
                env.render()
                time.sleep(0.0001)
            t += 1

        r = {
            'observations': np.array(obss_list)[:-1],
            'actions': np.array(acts_list),
            'rewards': np.array(rews_list),
            'latent': np.array(latent).reshape(1, -1),
            'velocity': np.mean([info['current_vel'] for info in info_list]),
        }
        rollouts.append(r)

        print(t, np.around(r['latent'], 2), np.sum(r['rewards']),
              np.mean(r['observations'][:, 3:7], axis=0),
              np.max(np.abs(r['actions'])),
              r['velocity'])

rets = [np.sum(r['rewards']) for r in rollouts]
print(f'Return Stats: mean {np.mean(rets)} std {np.std(rets)} min {np.min(rets)} max {np.max(rets)}')

# for i in range(len(rollouts)):
#     for j in range(i+1, len(rollouts)):
#         r0 = rollouts[i]['observations'].copy()
#         r1 = rollouts[j]['observations'].copy()

#         sd = np.linalg.norm(np.expand_dims(r0, 1) - np.expand_dims(r0, 0), axis=2)
#         cd = np.linalg.norm(np.expand_dims(r0, 1) - np.expand_dims(r1, 0), axis=2)

#         sd_tu = sd[np.triu_indices_from(sd, k=1)]
#         cd_tu = cd[np.triu_indices_from(sd, k=0)]

#         print(i, ': Self', sd_tu.min(), sd_tu.max())
#         print(i, j, ': Across', cd_tu.min(), cd_tu.max())
#         print(i, j, ': Across', np.sum((cd_tu - sd_tu.mean()) > 0) / cd_tu.shape[0])

DSTDIR = f"imitation_data/STRAT_h5"
os.makedirs(DSTDIR, exist_ok=True)

expert_states = np.concatenate([d['observations'] for d in rollouts], axis=0)
expert_actions = np.concatenate([d['actions'] for d in rollouts], axis=0)
expert_rewards = np.concatenate([d['rewards'] for d in rollouts], axis=0)
expert_masks_list = []
for d in rollouts:
    expert_masks_list.append(np.ones_like(d['observations'][:-1, 0]))
    expert_masks_list.append(np.zeros_like(d['observations'][-1:, 0]))
expert_masks = np.concatenate(expert_masks_list, axis=0)
expert_codes = np.concatenate([d['latent'] for d in rollouts], axis=0)
expert_ids = np.concatenate([np.ones_like(d['rewards'])*i for i, d in enumerate(rollouts)], axis=0)

# from matplotlib import pyplot as plt
# plt.figure()
# plt.figure(figsize=(5, 5))
# plt.xlim((-3, 3)); plt.ylim((-3, 3))
# for d in rollouts[::2]:
#     sxy = d['observations'] + np.random.randn(*d['observations'].shape)*0.0
#     plt.plot(sxy[:, 0], sxy[:, 1], 'k', linewidth=1)
# plt.tight_layout()
# plt.savefig(os.path.join(DSTDIR, f"{env_name}_vx.png"))

fname = f"{env_name}_vx.h5"
fpath = os.path.join(DSTDIR, fname)
assert not os.path.exists(fpath), fpath

hf = h5py.File(fpath, mode='w')
hf.create_dataset('expert_states', data=expert_states)
hf.create_dataset('expert_actions', data=expert_actions)
hf.create_dataset('expert_rewards', data=expert_rewards)
hf.create_dataset('expert_masks', data=expert_masks)
hf.create_dataset('expert_codes', data=expert_codes)
hf.create_dataset('expert_ids', data=expert_ids)

print(fpath, expert_actions.min(axis=0), expert_actions.max(axis=0))

# holder = dict()
# for r in rollouts:
#     if PENDULUM:
#         nlevel = np.clip(r["latent"], -0.49, 0.49)
#         nlevel = np.floor(nlevel*10)/10 + 0
#     if MAZE:
#         nlevel = np.round(r["latent"], 1) + 0
#     if HOPPER:
#         nlevel = np.clip(r["latent"], -0.49, 0.49)
#         nlevel = np.floor(nlevel*10)/10 + 0

#     key = f'{nlevel:0.2f}'
#     if key not in holder.keys():
#         holder[key] = []
#     holder[key].append(r)

# print({(k, len(v)) for k, v in holder.items()})

# DSTDIR = f"imitation_data/TRAJ_h5/{env_name}"
# os.makedirs(DSTDIR, exist_ok=True)

# for key, demos in holder.items():
#     num_states = np.sum([d['observations'].shape[0] for d in demos])
#     num_actions = np.sum([d['actions'].shape[0] for d in demos])
#     assert num_states == num_actions, (num_states, num_actions)

#     expert_states = np.concatenate([d['observations'] for d in demos], axis=0)
#     expert_actions = np.concatenate([d['actions'] for d in demos], axis=0)
#     expert_rewards = np.concatenate([np.zeros_like(d['observations'][:, 0]) for d in demos], axis=0)

#     expert_masks_list = []
#     for d in demos:
#         expert_masks_list.append(np.ones_like(d['observations'][:-1, 0]))
#         expert_masks_list.append(np.zeros_like(d['observations'][-1:, 0]))
#     expert_masks = np.concatenate(expert_masks_list, axis=0)

#     fname = f"{env_name}_TRAJ-N{num_states}_normal{key}.h5"
#     fpath = os.path.join(DSTDIR, fname)
#     assert not os.path.exists(fpath)

#     hf = h5py.File(fpath, mode='w')
#     hf.create_dataset('expert_states', data=expert_states)
#     hf.create_dataset('expert_actions', data=expert_actions)
#     hf.create_dataset('expert_rewards', data=expert_rewards)
#     hf.create_dataset('expert_masks', data=expert_masks)

#     print(fpath)
#     print(expert_actions.min(axis=0), expert_actions.max(axis=0))
