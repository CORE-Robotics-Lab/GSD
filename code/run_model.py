from my_utils import *
from args_parser import * 
from core.dqn import *
from core.ac import *
from core.irl import *
from core.vild import *
from vild_main import make_env
from core.agent import calc_mvn_params
from core.agent import chunks
from core.agent import kl_mvn as kl_metric
# from core.agent import logkl_mvn as kl_metric
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
os.environ['PYTHONWARNINGS'] = 'ignore'

def run_rollouts(args, env, policy_updater, latent_code_onehot_test, n_test_episodes, RENDER_FRAMES=False, LOGPEREP=False):

    ep_return_list = []
    rollout_list = []
    for i_episode in range(0, n_test_episodes):
        state = env.reset()

        if args.render and args.env_robosuite:
            env.viewer.set_camera(0)

        state_list = [state.copy()]
        action_list = []
        reward_list = []
        info_list = []
        frame_list = []
        reward_episode = 0
        for t in range(0, args.t_max):                   
            state_var = torch.FloatTensor(state)

            if latent_code_onehot_test is not None:
                state_var = torch.cat((state_var, latent_code_onehot_test), 0)  # input of the policy function. 
                
            action = policy_updater.greedy_action(state_var.unsqueeze(0)).to(device_cpu).detach().numpy()
            # next_state, reward, done, info = env.step(np.clip(action, a_min=a_low, a_max=a_bound) )    #same clip condition as expert trajectory
            next_state, reward, done, info = env.step(action)
            state = next_state
            reward_episode += reward
            action_list.append(action.copy())
            state_list.append(state.copy())
            reward_list.append(reward)
            info_list.append(info)
            frame_list.append(env.render('rgb_array', width=250, height=200) if RENDER_FRAMES else None)

            if args.render:
                env.render()
                time.sleep(0.0001)

            if t + 1 == args.t_max:
                done = 1
            if done : 
                break

        ep_return_list += [reward_episode]
        # ep_return_list += [reward]
        rollout_list.append({
            'observations': np.stack(state_list, axis=0),
            'actions': np.stack(action_list, axis=0),
            'rewards': np.array(reward_list),
            'infos': info_list,
            'frames': frame_list,
        })
        if LOGPEREP:
            print('Test epi %d: steps %d, return %.2f' % (i_episode, t, reward_episode))
            # print('\t\t\t\t\t', np.around(np.mean(state_list, axis=0), 3))

    return ep_return_list, rollout_list

def get_best_al(dset, zsvs, policy_updater, norm_fn, LOGPERDEMO):
    lps_dset = []
    for did, (state, action, _) in enumerate(dset):
        state_th = torch.Tensor(norm_fn(state))
        action_th = torch.Tensor(action)
        lps = []
        for zsv in zsvs:
            with torch.no_grad():
                zv_th = torch.Tensor(np.tile(np.expand_dims(zsv['zv'], axis=0), (state.shape[0], 1)))
                statez_th = torch.cat([state_th, zv_th], axis=1)
                lp = policy_updater.policy_net.get_log_prob(statez_th, action_th)
                lp = torch.mean(lp).item()
            lps.append(lp)
        idx = np.argmax(lps)
        # idx = np.argmin(lps)
        if LOGPERDEMO: print(did, idx, zsvs[idx]['zv'], lps[idx])
        lps_dset.append((lps[idx], zsvs[idx]["return"]))
    return lps_dset

def get_best_kl(dset, zsvs, include_action=False, LOGPERDEMO=False):
    lps_dset = []
    for did, (state, action, _) in enumerate(dset):
        dsv = np.concatenate([state] + ([action] if include_action else []), axis=1)
        dsv_mean = np.mean(dsv, axis=0)
        dsv_cov = np.cov(dsv, rowvar=False) + np.eye(dsv.shape[1])*1e-8
        lps = []
        for zsv in zsvs:
            zv_mean = zsv['sav_mean'] if include_action else zsv['sv_mean']
            zv_cov = zsv['sav_cov'] if include_action else zsv['sv_cov']
            # lp = np.linalg.norm(dsv - zv_mean)
            # lp = mvn.pdf(dsv, mean=zv_mean, cov=zv_cov)
            # lp = kl_metric((zv_mean, zv_cov), (dsv_mean, dsv_cov))
            lp = kl_metric((dsv_mean, dsv_cov), (zv_mean, zv_cov))
            # lp = js_mvn((zv_mean, zv_cov), (dsv_mean, dsv_cov))
            lps.append(lp)
        # idx = np.argmax(lps)
        idx = np.argmin(lps)
        z_str = np.around(zsvs[idx]["zv"].data.numpy(), 3)
        if LOGPERDEMO: print(f'DID: {did} SAMP {idx:3d} z:{z_str} \t\t{lps[idx]}')
        lps_dset.append((lps[idx], zsvs[idx]["return"]))
    return lps_dset

def init(args):

    # set args from ckptpath string and rerun default args
    # example ckptpath
    # 0_traj_type43_N5000-INFOGSD_PPO-100-100-relu_ec0.00010_gp0.010_cr5_ds5_bcs1000_bbs10000_es=n_dlt=disc_sr1_rc1_s1
    # /05.19_04.01.38
    # /models
    # /ckpt_policy_T5000000.pt

    _pths = glob.glob(args.ckptpath, recursive=True)
    assert len(_pths) == 1
    args.ckptpath = _pths[0]
    print(args.ckptpath)
    exp_name = args.ckptpath.lower().split('/')[-4]

    idxtrajname = exp_name.lower().split('-')[0]
    train_c_data = int(idxtrajname.split('type')[-1].split('_')[0])
    if args.c_data == 7:
        train_c_data_rem = train_c_data % 10
        if train_c_data_rem >= 3:
            train_c_data_rem -= 3
        args.c_data = train_c_data_rem
    print(exp_name)
    algohyps = '-'.join(exp_name.lower().split('-')[1:])
    hsizeactv = algohyps.split('_')[1].split('-')
    args.hidden_size = [int(x) for x in hsizeactv[1:3]]
    args.activation = hsizeactv[3]

    method_name_toks = exp_name.lower().split('-')[1].split('_')
    args.il_method, args.rl_method = method_name_toks[:2]
    if len(method_name_toks) == 3 and 'linear' == method_name_toks[2]:
        args.info_loss_type = 'linear'

    if "info" in args.il_method and args.encode_sampling is None:
        if "es=u" in exp_name:
            args.encode_sampling = "uniform"
        if "es=n" in exp_name:
            args.encode_sampling = "normal"
        if "es=c" in exp_name:
            args.encode_sampling = "cyclic"
    args.norm_obs = 0
    if "no1" in exp_name:
        args.norm_obs = 1
    if "no2" in exp_name:
        args.norm_obs = 2

    # defaults
    args.nthreads = 1

    args.normalize_code = 0
    if args.il_method == 'infogsd':
        args.normalize_code = 1
    
    if args.encode_sampling is None:
        # args.encode_sampling = 'uniform'
        args.encode_sampling = 'normal'

    print('v_data:', args.v_data)
    print('Inferred from ckptpath name:')
    print('il_method:', args.il_method)
    print('rl_method:', args.rl_method)
    print('activation:', args.activation)
    print('hidden_size:', args.hidden_size)
    print('norm_obs:', args.norm_obs)
    print('info_loss_type:', args.info_loss_type)
    print('encode_sampling:', args.encode_sampling)
    print('normalize_code:', args.normalize_code)
    print('tl_emb:', args.tl_emb)
    print('TRAINED C_DATA:', train_c_data)
    print('ARG C_DATA:', args.c_data)
    rl_default_parser(args)
    irl_default_parser(args)

    if args.nthreads is not None:
        torch.set_num_threads(args.nthreads)
    torch.manual_seed(args.seed)
    if use_gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        print(colored("Using CUDA.", p_color))
    np.random.seed(args.seed)
    random.seed(args.seed)
    test_cpu = True      # Set to True to avoid moving gym's state to gpu tensor every step during testing.

    env_name = args.env_name
    env, env_test, state_dim, action_dim, a_bound, is_disc_action, action_num = make_env(args)
    args.t_max = 1000

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # infer encode_dim from ckpt
    _wts = torch.load(args.ckptpath)
    args.encode_dim = encode_dim = _wts['affine_layers.0.weight'].size(1) - state_dim
    print('args.encode_dim:', args.encode_dim)

    """define actor and critic"""
    assert not is_disc_action
    if args.rl_method == "ac":
        policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
    if args.rl_method == "sac":
        policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
    if args.rl_method == "td3":
        policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
    if args.rl_method == "trpo":
        policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
    if args.rl_method == "ppo" or args.rl_method == "ppobc":
        policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
    # class DemoAgent(object):
    #     def __init__(self):
    #         self.dummy = IRL(1, 1, args, initialize_net=True, load_verbose=False)
    #         self.update_type = "on_policy"
    #         self.i = -1
    #         self.did = 0
    #         self.load_model = lambda x: None
    #         self.policy_to_device = lambda x: None
    #     def greedy_action(self, *args, **kwargs):
    #         self.i += 1
    #         try:
    #             action = self.dummy.expert_actions_train[self.did][self.i]
    #         except:
    #             self.i = -1
    #             self.did += 1
    #             action = np.array([0, 0, 0, 0])
    #         return torch.FloatTensor(action)
    # policy_updater = DemoAgent()

    update_type = policy_updater.update_type  # "on_policy" or "off_policy"
    if args.max_step is None:
        if update_type == "on_policy":
            args.max_step = 5000000
            if args.psi_param_std is None: args.psi_param_std = 0 
        elif update_type == "off_policy":
            args.max_step = 1000000     
            if args.psi_param_std is None: args.psi_param_std = 1 
        if args.env_atari:
            args.max_step = args.max_step * 10 

        try:
            if args.il_method == "infogail":  
                discriminator_updater = InfoGAIL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # AIRL version need entropy coefficent and policy   
            elif args.il_method == "infogsd":  
                discriminator_updater = GSD(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)
            else:
                discriminator_updater = None
        except:
            discriminator_updater = None

    # load model from checkpoint
    model_filename = args.ckptpath
    policy_updater.load_model(model_filename)
    policy_updater.policy_to_device(device_cpu) 
    print("Policy model is loaded from %s" % model_filename )

    try:
        model_filename = args.ckptpath.replace('policy', 'discr')
        model_dict = torch.load(model_filename)
        discriminator_updater.posterior_net.load_state_dict(model_dict['posterior_net'])
        print("Discr model is loaded from %s" % model_filename )
    except:
        pass

    # n_test_episodes = 5 if not args.det_env else 1
    n_test_episodes = 1

    return args, env_test, policy_updater, n_test_episodes, discriminator_updater

def init_info(args, NZ):
    print(f'***** num zs = {NZ} *****')

    info_list = [None]
    if "info" in args.il_method:
        if not args.encode_cont:
            z_list = np.arange(0, args.encode_dim)
        else:
            # hardcode here
            # args.encode_sampling = 'normal'
            # args.encode_sampling = 'uniform'
            # args.encode_sampling = 'cyclic'
            # args.encode_sampling = 'spaced'
            print(f'***** args.encode_sampling = {args.encode_sampling} *****')

            if args.encode_sampling == 'normal' or args.encode_sampling == 'archive':
                z_list = np.random.normal(0, 1, size=(NZ, args.encode_dim))
            elif args.encode_sampling == 'uniform':
                z_list = np.random.uniform(-1, 1, size=(NZ, args.encode_dim))
            elif args.encode_sampling == 'spaced':
                range_bound = 2.5
                num_points = 10
                z0s = np.linspace(-range_bound, range_bound, num=num_points)
                z1s = np.linspace(-range_bound, range_bound, num=num_points)
                z_list = np.array([[z0, z1] for z0 in z0s for z1 in z1s])
                # z_list = np.array([[z0, z1] for z1 in z1s for z0 in z0s])
            else:
                raise NotImplementedError
            if args.normalize_code:
                z_list = z_list / (np.linalg.norm(z_list, axis=1, keepdims=True) + 1e-8)

        info_list = []
        for z_i in z_list:
            if not args.encode_cont:
                latent_code_test = torch.LongTensor(1, 1).fill_(z_i)
                latent_code_onehot_test = torch.FloatTensor(1, args.encode_dim)
                latent_code_onehot_test.zero_()
                latent_code_onehot_test.scatter_(1, latent_code_test, 1)
                latent_code_onehot_test = latent_code_onehot_test.squeeze()  #should have size [num_worker]
            else:
                latent_code_onehot_test = torch.FloatTensor(z_i)
            info_list.append(latent_code_onehot_test)

    return info_list

def visualize(args, test_seed=1):
    num_info = args.num_info if args.num_info is not None else 50
    args, env, policy_updater, n_test_episodes, disc_updater = init(args)
    info_list = init_info(args, num_info)

    if 'hc' in args.env_name.lower():
        MKEY = 'current_vel'
        # n_test_episodes = 1
        n_test_episodes = 5
    elif 'pend' in args.env_name.lower():
        MKEY = 'ramp_pos'
        n_test_episodes = 1
    elif 'fetch' in args.env_name.lower():
        MKEY = 'obj_pos_y'
        n_test_episodes = 1
    else:
        raise NotImplementedError(args.env_name)

    test_list_all = []
    rollout_list_all = []

    # info_list = [
    #     # [ 0.9813208,  0.51421887],
    #     [-1.1880176, -0.5497462 ],
    # ]
    info_list = [torch.Tensor(x) for x in info_list]
    for info_i in tqdm(info_list):
        if use_gpu:
            torch.cuda.manual_seed(test_seed)
            torch.backends.cudnn.deterministic = True
        np.random.seed(test_seed)
        random.seed(test_seed)

        latent_code_onehot_test = info_i
        if 'Pend' in args.env_name:
            args.t_max = 100
        ep_return_list, rollout_list = run_rollouts(args, env, policy_updater, latent_code_onehot_test, n_test_episodes, RENDER_FRAMES=False, LOGPEREP=True)
        ep_return_list = np.asarray(ep_return_list)
        print('Test model with Z: %s Average return: %.2f' % (info_i, ep_return_list.mean()))

        for i_r in range(len(rollout_list)):
            rollout_list[i_r][MKEY+'_mean'] = np.mean([info_dict[MKEY] for info_dict in rollout_list[i_r]['infos']])

        vels_per_z = np.array([r_i[MKEY+'_mean'] for r_i in rollout_list])
        print('Vels:', vels_per_z)

        test_list_all.append(ep_return_list.mean())
        rollout_list_all.append(rollout_list)

    print(np.mean(test_list_all), np.std(test_list_all), np.min(test_list_all), np.max(test_list_all))

    codes = np.array([it.data.numpy() for it in info_list])
    # vels = np.array([np.mean([r_i['traj_mean_vel'] for r_i in rlist]) for rlist in rollout_list_all])
    vels = np.array([[r_i[MKEY+'_mean'] for r_i in rlist] for rlist in rollout_list_all])
    # print(vels)
    print('STD:', np.mean(np.std(vels, axis=1)))

    # import imageio
    # for ri, rl in enumerate(rollout_list_all[0]):
    #     # imageio.mimwrite(f'results_IL/{ri}.gif', rl['frames'], fps=200)
    #     # imageio.mimsave(f'results_IL/{ri}.gif', rl['frames'], format= 'GIF', fps=200)
    #     imageio.mimsave(f'results_IL/{ri}.gif', rl['frames'], format= 'GIF', duration=1000/200)

    # tgtvel = 3.65
    # dists = np.square(vels - tgtvel)
    # didxs = np.argsort(dists)[:5]
    # tgtcodes = codes[didxs]
    # print(tgtcodes, dists[didxs])

    # print(f'Displayed: 3D plot of vels for {args.env_name}')
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter3D(codes[:, 0], codes[:, 1], np.std(vels, axis=1))
    # plt.show()

def plot_demos(args):
    rl_default_parser(args)
    irl_default_parser(args)
    # args = init(args)[0]
    dummy = IRL(1, 1, args, initialize_net=True, load_verbose=False)
    dset_train = list(zip(dummy.expert_states_train, dummy.expert_actions_train, range(len(dummy.expert_states_train))))
    dset_val = list(zip(dummy.expert_states_val, dummy.expert_actions_val, range(len(dummy.expert_states_val))))
    dset_test = list(zip(dummy.expert_states_test, dummy.expert_actions_test, range(len(dummy.expert_states_train))))

    from matplotlib import pyplot as plt
    plt.figure(figsize=(5, 10))
    plt.subplot(2, 1, 1)
    plt.xlim((-3, 3)); plt.ylim((-3, 3))
    for ss, ac, did in dset_train:
        sxy = ss[:, :2] + np.random.randn(ss.shape[0], 2)*0.0
        plt.plot(sxy[:, 0], sxy[:, 1], 'k', linewidth=1)
    plt.subplot(2, 1, 2)
    plt.xlim((-3, 3)); plt.ylim((-3, 3))
    for ss, ac, did in dset_test:
        sxy = ss[:, :2] + np.random.randn(ss.shape[0], 2)*0.0
        plt.plot(sxy[:, 0], sxy[:, 1], 'k', linewidth=1)
    plt.tight_layout()
    plt.savefig("demos_0.png")

def evaluate(args, test_seed=1, robustness=True, LOGPERDEMO=True):
    # set num z to be sampled here
    num_info = args.num_info
    args, env, policy_updater, n_test_episodes, disc_updater = init(args)
    # n_test_episodes = 1
    # n_test_episodes = 5
    n_test_episodes = args.num_eps
    NPARALLEL = 2

    # to ensure different codes are sampled
    torch.manual_seed(test_seed)
    np.random.seed(test_seed)
    random.seed(test_seed)

    print(f'***** Max Ep Steps: {args.t_max} args.seed {args.seed} test_seed {test_seed} *****')
    print(f'*** budgets = {args.bgt_info} NPARALLEL = {NPARALLEL} n_test_episodes = {n_test_episodes} ***')

    # sampled uniformly from prior
    info_list = init_info(args, num_info)
    test_seed_list = [(test_seed+idx)*54 for idx in range(len(info_list))]

    if 'hc' in args.env_name.lower():
        MKEY = 'current_vel'
        # MTGTVALS = np.linspace(1, 5, 9)
        # MTGTVALS = [1.78, 3.65]
        # MTGTVALS = [0.97, 2.88, 4.87]
        # MTGTVALS = [1.5, 2.0, 2.5, 3.5, 4.0, 4.5]
        MTGTVALS = [0.97, 1.78, 2.88, 3.65, 4.87]
    elif 'pend' in args.env_name.lower():
        MKEY = 'ramp_pos'
        MTGTVALS = [-0.4, -0.2, 0.0, 0.2, 0.4]
        if args.v_data == 2:
            MTGTVALS = [-0.721, -0.374, 0.0, 0.407, 0.741]
    elif 'fetch' in args.env_name.lower():
        MKEY = 'obj_pos_y'
        MTGTVALS = np.linspace(-0.8, 0.8, 5)*0.2 + 0.75
        if 'wide' in args.env_name.lower():
            MTGTVALS = np.linspace(-0.8, 0.8, 5)*0.4 + 0.75
    else:
        raise NotImplementedError(args.env_name)
    print(MKEY, MTGTVALS)

    info_list_chunks = chunks(info_list, NPARALLEL)
    test_seed_chunks = chunks(test_seed_list, NPARALLEL)
    env_test_chunks = [make_env(args)[1] for _ in range(NPARALLEL)]

    def sample_from_env(idx):
        zsvs_ret = []
        for test_seed_i, info_i in zip(test_seed_chunks[idx], tqdm(info_list_chunks[idx], disable=idx!=0)):
            np.random.seed(test_seed_i)
            random.seed(test_seed_i)
            env_test_chunks[idx].seed(test_seed_i)

            latent_code_onehot_test = info_i
            ep_return_list, rollout_list = run_rollouts(
                args, env_test_chunks[idx],
                policy_updater,
                latent_code_onehot_test,
                n_test_episodes)
            ep_return_list = np.asarray(ep_return_list)
            # print('Test model with Z: %s Average return: %.2f' % (info_i, ep_return_list.mean()))
            # print([np.mean([info[MKEY] for info in r['infos']]) for r in rollout_list])

            if args.norm_obs:
                for i in range(len(rollout_list)):
                    rollout_list[i]['observations'] = env.unnorm_obs(rollout_list[i]['observations'])

            sv = []
            sav = []
            for r in rollout_list:
                sv.append(r['observations'])
                sav.append(np.concatenate([
                    r['observations'][:-1],
                    r['actions'],
                ], axis=1))
            sv = np.concatenate(sv, axis=0)
            sav = np.concatenate(sav, axis=0)

            zsvs_ret.append({
                'zv': info_i,
                'return': ep_return_list.mean(),
                'lastrew': np.mean([r['rewards'][-1] for r in rollout_list]),
                'sv_mean': np.mean(sv, axis=0),
                'sv_cov': np.cov(sv, rowvar=False) + np.eye(sv.shape[1])*1e-8,
                'sav_mean': np.mean(sav, axis=0),
                'sav_cov': np.cov(sav, rowvar=False) + np.eye(sav.shape[1])*1e-8,
                MKEY+'_mean': np.mean([np.mean([info[MKEY] for info in r['infos']]) for r in rollout_list]),
                MKEY+'_std': np.std([np.mean([info[MKEY] for info in r['infos']]) for r in rollout_list]),
            })
        return zsvs_ret

    # chunked + serial
    # zsvs = [sample_from_env(pidx) for pidx in range(NPARALLEL)]
    # chunked + parallel
    zsvs = Parallel(n_jobs=NPARALLEL)(delayed(sample_from_env)(pidx) for pidx in range(NPARALLEL))
    # make into single list
    zsvs = [pitem for plist in zsvs for pitem in plist]
    zvs = np.array([x.data.numpy() for x in info_list])
    print("zvs shape", zvs.shape[0])

    rets = np.array([r['return'] for r in zsvs])
    print('RT', np.mean(rets), np.std(rets), np.max(rets), np.min(rets))
    lrws = np.array([r['lastrew'] for r in zsvs])
    print('LR', np.mean(lrws), np.std(lrws), np.max(lrws), np.min(lrws))

    # test vel recovery
    vstds = np.array([r[MKEY+'_std'] for r in zsvs])
    vstd_all = np.sqrt(np.mean(np.square(vstds)))
    print(f'VL-std-all', vstd_all, 0.0)

    vels = np.array([r[MKEY+'_mean'] for r in zsvs])
    codes = zvs.copy()

    budget_s = args.bgt_info
    followuprolls = 0

    for budget in budget_s:
        print(f'*** budget = {budget}')
        codes_chunks = chunks(codes, args.num_info // budget)
        vels_chunks = chunks(vels, args.num_info // budget)
        rets_chunks = chunks(rets, args.num_info // budget)

        errors_all = []
        rets_all = []
        for it, tgtvel in enumerate(tqdm(MTGTVALS, disable=True)):
            errors_split = []
            rets_split = []
            for code_chunk, vel_chunk, ret_chunk in zip(codes_chunks, vels_chunks, rets_chunks):
                dist_chunk = np.abs(vel_chunk - tgtvel)
                didxs_chunk = np.argsort(dist_chunk)
                tgtcodes = code_chunk[didxs_chunk[:1]]
                # tgtcodes = [[-1.1880176, -0.5497462 ]]

                # use same traj
                testvels = vel_chunk[didxs_chunk[:1]]
                testrets = ret_chunk[didxs_chunk[:1]]
                # sample 5 traj with latent
                # _, veltestrolls = run_rollouts(args, env, policy_updater, torch.FloatTensor(tgtcodes[0]), followuprolls)
                # testvels = np.array([np.mean([ifdict['current_vel'] for ifdict in ri['infos']]) for ri in veltestrolls])

                # print(tgtvel, tgtcodes[0], testvels, testrets)
                tgtvelerrors = np.abs(testvels - tgtvel)
                errors_split.append(tgtvelerrors)
                rets_split.append(testrets)
            errors_split = np.concatenate(errors_split, axis=0)
            rets_split = np.concatenate(rets_split, axis=0)

            print(f'VL{budget}-{it+1}', np.mean(errors_split), np.std(errors_split))
            print(f'RT-VL{budget}-{it+1}', np.mean(rets_split), np.std(rets_split), np.max(rets_split), np.min(rets_split))
            errors_all.append(np.mean(errors_split))
            rets_all.append(np.mean(rets_split))

        print(f'VL{budget}-all', np.mean(errors_all), np.std(errors_all))
        print(f'RT-VL{budget}-all', np.mean(rets_all), np.std(rets_all), np.max(rets_all), np.min(rets_all))

if __name__ == "__main__":
    args = args_parser(set_defaults=False)
    if args.mode == 'viz':
        visualize(args, test_seed=args.test_seed)
    elif args.mode == 'prior':
        evaluate(args, robustness=False, test_seed=args.test_seed, LOGPERDEMO=False)
    else:
        raise NotImplementedError
