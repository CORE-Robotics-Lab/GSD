from my_utils import *
from args_parser import * 
from core.dqn import *
from core.ac import *
from core.irl import *
from core.vild import *
from vild_main import make_env
from core.agent import calc_mvn_params
from core.agent import kl_mvn as kl_metric
# from core.agent import logkl_mvn as kl_metric
from tqdm import tqdm
import glob
from joblib import Parallel, delayed
os.environ['PYTHONWARNINGS'] = 'ignore'

def run_rollouts(args, env, policy_updater, latent_code_onehot_test, n_test_episodes, LOGPEREP=False):

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
        })
        if LOGPEREP:
            print('Test epi %d: steps %d, return %.2f' % (i_episode, t, reward_episode))
            # print('\t\t\t\t\t', np.around(np.mean(state_list, axis=0), 3))
            metrics_keys = rollout_list[0]['infos'][0].keys()
            metrics_vals = {}
            for mk in metrics_keys:
                metrics_vals[f'{mk}/mean'] = np.mean([info_dict[mk] for info_dict in rollout_list[0]['infos']])
                metrics_vals[f'{mk}/std'] = np.std([info_dict[mk] for info_dict in rollout_list[0]['infos']])
            print(metrics_vals)

    return ep_return_list, rollout_list

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
    exp_name = exp_name.lower().replace('-', '_')
    print(exp_name)

    algocfgs = exp_name.split('_')
    args.il_method, args.rl_method = None, algocfgs[1]
    args.hidden_size = [int(x) for x in algocfgs[2:4]]
    args.activation = algocfgs[4]

    print('Inferred from ckptpath name:')
    print('il_method:', args.il_method)
    print('rl_method:', args.rl_method)
    print('activation:', args.activation)
    print('hidden_size:', args.hidden_size)
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

        if args.il_method == "infogail":  
            discriminator_updater = InfoGAIL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # AIRL version need entropy coefficent and policy   
        elif args.il_method == "infogsd":  
            discriminator_updater = GSD(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)
        else:
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

def visualize(args, test_seed=1):
    args, env, policy_updater, n_test_episodes, disc_updater = init(args)

    test_list_all = []

    # info_list = [torch.FloatTensor([-1.5, 0.75])]
    # info_list = [
        # [1.1666666667, -1.5000],
        # [ 1.1666666667, -1.1666666667],
        # [ 1.16666666666666667, -0.8333333333333333],
        # [ 1 + 1/6, -1 + 1/6],
        # [ 1.1666666667, -0.5000],
        # [ 1.1666666667, -0.1666666667],
        # [ 1.5000, -0.8333333333],
        # [ 1.5000, -0.1666666667],
        # [ 0.8333333333, -1.5000],
        # [-0.5164,  0.8463, -0.1309],
    # ]
    for idx in tqdm(range(10)):
        if use_gpu:
            torch.cuda.manual_seed(test_seed)
            torch.backends.cudnn.deterministic = True
        np.random.seed(test_seed)
        random.seed(test_seed)

        if 'Pend' in args.env_name:
            args.t_max = 100
        ep_return_list, _ = run_rollouts(args, env, policy_updater, None, n_test_episodes, LOGPEREP=True)
        ep_return_list = np.asarray(ep_return_list)
        print('Test model Average return: %.2f' % (ep_return_list.mean()))

        test_list_all.extend(ep_return_list)

    print(np.mean(test_list_all), np.std(test_list_all))

if __name__ == "__main__":
    args = args_parser(set_defaults=False)
    # plot_demos(args)
    if args.mode == 'viz':
        visualize(args, test_seed=1)
    else:
        raise NotImplementedError
