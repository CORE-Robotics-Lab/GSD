
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
env = make_env(args)[0]
demo_np_random = np.random.RandomState(args.seed+42)

RENDER = args.render
assert 'Fetch' in env_name

def demo_pickplace_policy(obs, state, latent):
    if state is None:
        # obj1_dlt = np.array([0.25, latent*0.2])
        # changed for wide env
        obj1_dlt = np.array([0.20, latent*0.4])
        state = {
            'obj1_dlt': obj1_dlt,
            'latent': latent,
            'callstack': ['main'],
        }
    action = np.array([0.0, 0.0, 0.0, 0.0])

    if len(state['callstack']) == 0:
        return action, state

    grip_pos = obs[0:3]
    obj1_pos = obs[3:6]
    obj1_vel = obs[6:9]
    grip_vel = obs[9:12]
    grip_wdt = obs[12]
    if 'obj1_tgt' not in state:
        obj1_dlt = state['obj1_dlt']
        obj1_tgt = obj1_pos.copy()
        obj1_tgt[:2] = obj1_pos[:2] + state['obj1_dlt']
        state['obj1_tgt'] = obj1_tgt
    obj1_tgt = state['obj1_tgt']
    grip_tgt = None
    grip_del = np.zeros(3)
    grip_ref = None

    if state['callstack'][0] == 'pdb':
        import pdb; pdb.set_trace()

    elif state['callstack'][0] == 'open':
        if grip_wdt < 0.05:
            action[3] = 1.0
        else:
            state['callstack'].pop(0)

    elif state['callstack'][0] == 'close':
        if grip_wdt > 0.04:
            action[3] = -1.0
        else:
            state['callstack'].pop(0)

    elif state['callstack'][0].split('_')[0].startswith('goto'):
        gototoks = state['callstack'][0].split('_')
        if len(gototoks) == 2:
            keyt, objt = state['callstack'][0].split('_')
            grpt = 'free'
        else:
            keyt, grpt, objt = state['callstack'][0].split('_')

        grip_tgt = grip_tgt if grip_tgt is not None else (obj1_pos if objt == '1c' else None)
        grip_tgt = grip_tgt if grip_tgt is not None else (obj1_tgt if objt == '1t' else None)
        # fetch
        grip_tgt = grip_tgt if grip_tgt is not None else ([0, 0, 0.6] if objt == 'hh' else None)
        grip_tgt = grip_tgt if grip_tgt is not None else ([0, 0, 0.5] if objt == 'll' else None)
        # print(objt, grip_tgt)
        if grpt == 'hold':
            action[3] = -1.0
        if keyt.endswith('xy'):
            grip_del = grip_tgt[:2] - grip_pos[:2]
            action[:2] = 10*grip_del
        if keyt.endswith('z'):
            grip_del = grip_tgt[2:3] - grip_pos[2:3]
            action[2:3] = 10*grip_del
        if np.linalg.norm(grip_del) < 0.01:
            state['callstack'].pop(0)

    elif state['callstack'][0].split('_')[0].startswith('place'):
        _, objt = state['callstack'][0].split('_')
        state['callstack'].pop(0)
        state['callstack'].insert(0, f'gotoz_hold_{objt}')
        state['callstack'].insert(0, f'gotoxy_hold_{objt}')
        state['callstack'].insert(0, f'gotoz_hh')

    elif state['callstack'][0].split('_')[0].startswith('pick'):
        _, objt = state['callstack'][0].split('_')
        state['callstack'].pop(0)
        state['callstack'].insert(0, f'close')
        state['callstack'].insert(0, f'gotoz_{objt}')
        state['callstack'].insert(0, f'open')
        state['callstack'].insert(0, f'gotoxy_{objt}')
        state['callstack'].insert(0, f'gotoz_hh')

    elif state['callstack'][0] == 'main':
        state['callstack'].pop(0)

        # push blocks to certain location
        state['callstack'].insert(0, 'place_1t')
        state['callstack'].insert(0, 'gotoz_hold_hh')
        state['callstack'].insert(0, 'pick_1c')

    # print(state['callstack'])
    # print('obj1', obj1_pos, '->', obj1_tgt)

    action = np.clip(action, -0.5, 0.5)
    return action, state

# latent_values = np.linspace(-1, 1, 5)
# latent_values = np.linspace(-1, 1, 25)
# latent_values = np.linspace(-1, 1, 51)
tiled_latents = np.tile(np.linspace(-0.8, 0.8, 5).reshape(5, 1), (1, 5))
noise_latents = np.array([np.sort(x) for x in np.random.randn(5, 5)])
# latent_values = (tiled_latents + noise_latents*0.065).reshape(-1)
latent_values = (tiled_latents + noise_latents*0.05).reshape(-1)
n_episodes = 1

rollouts = []
# frames = []
for latent in latent_values:

    for i in range(n_episodes):
        obs, done = env.reset(), False
        obss_list = [obs.copy()]
        acts_list = []
        rews_list = []

        # xx
        polstate = None

        t = 0
        while not done:

            # action = np.array([0, 0, 0, 1])
            action, polstate = demo_pickplace_policy(obs, polstate, latent)
            action[:3] += demo_np_random.randn(*action[:3].shape)*0.02
            # action += np.random.randn(*action.shape)*0.1
            # print(action)

            obs, rew, done, info = env.step(action)

            obss_list.append(obs.copy())
            acts_list.append(action.copy())
            # rews_list.append(rew)
            rews_list.append(0 if not done else rew)

            if RENDER:
                env.render()
                time.sleep(0.0001)
            # frames.append(env.render(mode='rgb_array'))
            t += 1

        r = {
            'observations': np.array(obss_list)[:-1],
            'actions': np.array(acts_list),
            'rewards': np.array(rews_list),
            'latent': np.array(latent).reshape(1, -1),
        }
        rollouts.append(r)

        print(t, np.around(r['latent'], 2), np.sum(r['rewards']),
              np.mean(r['observations'][:, 0:6], axis=0),
              np.max(np.abs(r['actions'])))

rets = [np.sum(r['rewards']) for r in rollouts]
print(f'Return Stats: mean {np.mean(rets)} std {np.std(rets)} min {np.min(rets)} max {np.max(rets)}')
assert np.min(rets) > -0.02, 'Bad demos'

DSTDIR = f"imitation_data/STRAT_h5"
os.makedirs(DSTDIR, exist_ok=True)

# import imageio
# imageio.mimsave(os.path.join(DSTDIR, 'fetch_stack.gif'), frames[::4])
# imageio.mimsave(os.path.join(DSTDIR, 'fetch_double_push.gif'), frames[::4])
# imageio.mimsave(os.path.join(DSTDIR, 'fetch_single_push.gif'), frames[::4])
# imageio.mimsave(os.path.join(DSTDIR, 'fetch_pick_place.gif'), frames[::4])

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

fname = f"{env_name}_vx.h5"
fpath = os.path.join(DSTDIR, fname)
assert not os.path.exists(fpath)

hf = h5py.File(fpath, mode='w')
hf.create_dataset('expert_states', data=expert_states)
hf.create_dataset('expert_actions', data=expert_actions)
hf.create_dataset('expert_rewards', data=expert_rewards)
hf.create_dataset('expert_masks', data=expert_masks)
hf.create_dataset('expert_codes', data=expert_codes)
hf.create_dataset('expert_ids', data=expert_ids)

print(fpath, expert_actions.min(axis=0), expert_actions.max(axis=0))
