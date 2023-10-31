from my_utils import *
from my_utils.my_gym_utils import DetWrapper
from args_parser import * 
from core.agent import Agent
from datetime import datetime
import imageio
import importlib
import wandb

from core.dqn import *
from core.ac import *
from core.irl import *
from core.gsdr import *
from core.vild import *
from core.ac_bc import *

def make_env(args):
    env_name = args.env_name
    env, env_test = None, None
    state_dim, action_dim, a_bound = None, None, None
    is_disc_action, action_num = None, None
    """ Create environment and get environment's info. """
    if args.env_atari:
        from my_utils.atari_wrappers import Task 
        env = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
        env_test = Task(env_name, num_envs=1, clip_rewards=False, seed=args.seed)     
    elif args.env_bullet:
        import pybullet 
        import pybullet_envs 
        pybullet.connect(pybullet.DIRECT)
        env = gym.make(env_name)
        env.seed(args.seed)  
        env_test = env        
        if args.render:
            env_test.render(mode="human")
    elif args.env_robosuite:
        from my_utils.my_robosuite_utils import make_robosuite_env
        args.t_max = 500 
        env = make_robosuite_env(args)
        env_test = make_robosuite_env(args)
        # the sampler use functions from python's random, so the seed are already set.
        env_name = args.env_name + "_reach"
    elif args.env_custom:
        if 'Maze' in env_name:
            env_class = getattr(importlib.import_module('my_utils.my_envs'), env_name)
            if 'Long' in env_name or 'Cross' in env_name:
                env_horizon = 25
            elif 'Square' in env_name:
                env_horizon = 40
            else:
                env_horizon = 10
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env_class(), env_horizon)
            env_test = TimeLimit(env_class(), env_horizon)
        if 'InvertedPendulum' in env_name:
            env_kwargs = {}
            if 'Pos' in env_name:
                postok = env_name.split('Pos')[1]
                env_kwargs['target_pos'] = 0.8 * (2*(float(postok)-10)/(50-10) - 1)
                env_class = getattr(importlib.import_module('my_utils.my_envs'), 'InvertedPendulumCustomPos')
            else:
                env_class = getattr(importlib.import_module('my_utils.my_envs'), 'InvertedPendulumCustom')
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env_class(**env_kwargs), 1000)
            env_test = TimeLimit(env_class(**env_kwargs), 1000)
            if 'Det' in env_name:
                env = DetWrapper(env)
                env_test = DetWrapper(env_test)
        if 'HCCustom' in env_name:
            env_kwargs = {}
            if 'Vel' in env_name:
                base_name = 'HCCustomVel'
                env_kwargs['target_vel'] = float(env_name[len(base_name):])/10
                env_class = getattr(importlib.import_module('my_utils.my_cheetah_envs'), base_name)
            elif env_name == 'HCCustom':
                env_class = getattr(importlib.import_module('my_utils.my_cheetah_envs'), env_name)
            else:
                raise NotImplementedError
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env_class(**env_kwargs), 1000)
            env_test = TimeLimit(env_class(**env_kwargs), 1000)
        if 'HopperCustom' in env_name:
            from my_utils.my_envs import HopperCustom
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(HopperCustom(), 1000)
            env_test = TimeLimit(HopperCustom(), 1000)
        if 'HumanoidCustom' in env_name:
            from my_utils.my_envs import HumanoidCustom
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(HumanoidCustom(), 1000)
            env_test = TimeLimit(HumanoidCustom(), 1000)
        if 'WalkerCustom' in env_name:
            env_kwargs = {}
            if env_name == 'WalkerCustom':
                # from my_utils.my_walker_envs import WalkerCustom as env_class
                env_class = getattr(importlib.import_module('my_utils.my_walker_envs'), env_name)
            elif 'Short' in env_name:
                env_class = getattr(importlib.import_module('my_utils.my_walker_envs'), env_name)
            else:
                env_name_, env_param_ = env_name.split('-')
                env_class = getattr(importlib.import_module('my_utils.my_walker_more_envs'), env_name_)
                env_kwargs['param'] = int(env_param_)
            from gym.wrappers.time_limit import TimeLimit
            env = TimeLimit(env_class(**env_kwargs), 1000)
            env_test = TimeLimit(env_class(**env_kwargs), 1000)
        if 'Fetch' in env_name:
            FetchEnv = getattr(importlib.import_module('my_utils.my_fetch_envs'), env_name)
            from gym.wrappers.time_limit import TimeLimit
            from gym.wrappers import FilterObservation, FlattenObservation
            def _fetch_init():
                env = FetchEnv()
                env = FilterObservation(env, ['observation'])
                env = FlattenObservation(env)
                env = TimeLimit(env, 100)
                return env
            env = _fetch_init()
            env_test = _fetch_init()
        env.seed(args.seed)
        env_test.seed(args.seed)
    else: 
        env = gym.make(env_name)     
        env.seed(args.seed)  
        env_test = gym.make(env_name)
        env_test.seed(args.seed)  

    state_dim = env.observation_space.shape[0]
    is_disc_action = args.env_discrete
    action_dim = (0 if is_disc_action else env.action_space.shape[0])
    if args.env_robosuite:
        action_dim = action_dim - 1     # we disable gripper for reaching 
    if is_disc_action:
        a_bound = 1
        action_num = env.action_space.n 
        print("State dim: %d, action num: %d" % (state_dim, action_num))
    else:
        """ always normalize env. """ 
        asscalar = lambda x: x.item()
        if asscalar(env.action_space.high[0]) != 1:
            from my_utils.my_gym_utils import NormalizeGymWrapper
            env = NormalizeGymWrapper(env)
            env_test = NormalizeGymWrapper(env_test)
            print("Use action-normalized environments.")
        a_bound = asscalar(env.action_space.high[0])
        a_low = asscalar(env.action_space.low[0])
        assert a_bound == -a_low 
        assert a_bound == 1 
        print("State dim: %d, action dim: %d, action bound %d" % (state_dim, action_dim, a_bound))

        if "LunarLanderContinuous" in env_name or "BipedalWalker" in env_name:
            from my_utils.my_gym_utils import ClipGymWrapper
            env = ClipGymWrapper(env) 
            env_test = ClipGymWrapper(env_test) 

    if args.det_env:
        env = DetWrapper(env)
        env_test = DetWrapper(env_test)

    if args.norm_obs > 0:
        # normalize obs with fixed mean, std, based on datasets
        from my_utils.my_gym_utils import get_preset_norm_params, NormalizeObsWrapper
        obs_mean, obs_var = get_preset_norm_params(args.env_name)
        if args.norm_obs == 2:
            # ignore variance scaling for norm_obs = 2
            obs_var = np.ones_like(obs_var)
        env = NormalizeObsWrapper(env, obs_mean, obs_var)
        env_test = NormalizeObsWrapper(env_test, obs_mean, obs_var)
        print("Use state-normalized environments.")

    return env, env_test, state_dim, action_dim, a_bound, is_disc_action, action_num

""" The main entry function for RL """
def main(args, logger_fn=None):

    if args.il_method is None:
        method_type = "RL"  # means we just do RL with environment's rewards 
        info_method = False 
        encode_dim = 0 
    else:
        method_type = "IL"
        if "info" in args.il_method:
            info_method = True
            encode_dim = args.encode_dim
        else:
            info_method = False 
            encode_dim = 0 

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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """define actor and critic"""
    if is_disc_action:  # work in progress...
        if args.rl_method == "dqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=False, encode_dim=encode_dim)
        if args.rl_method == "ddqn":
            policy_updater = DQN(state_dim=state_dim, action_num=action_num, args=args, double_q=True, encode_dim=encode_dim)
        if args.rl_method == "qr_dqn":
            policy_updater = QR_DQN(state_dim=state_dim, action_num=action_num, args=args, encode_dim=encode_dim)
        if args.rl_method == "clipped_ddqn":
            policy_updater = Clipped_DDQN(state_dim=state_dim, action_num=action_num, args=args, encode_dim=encode_dim)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=action_num, is_discrete=True, encode_dim=encode_dim)
    else:
        if args.rl_method == "ac":
            policy_updater = AC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "sac":
            policy_updater = SAC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "td3":
            policy_updater = TD3(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "trpo":
            policy_updater = TRPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "ppo":
            policy_updater = PPO(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)
        if args.rl_method == "ppobc":
            policy_updater = PPOBC(state_dim=state_dim, action_dim=action_dim, args=args, a_bound=a_bound, encode_dim=encode_dim)

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
        
    if method_type == "IL":
        if args.il_method == "irl": # maximum entropy IRL
            discriminator_updater = IRL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "gail":
            discriminator_updater = GAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "vail":
            discriminator_updater = VAIL(state_dim=state_dim, action_dim=action_dim, args=args)
        elif args.il_method == "airl":
            discriminator_updater = AIRL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)  # need entropy coefficient and policy         
        elif args.il_method == "vild":  
            discriminator_updater = VILD(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # need entropy coefficient           
        elif args.il_method == "infogail":  
            discriminator_updater = InfoGAIL(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)   # AIRL version need entropy coefficent and policy   
        elif args.il_method == "infogsdr":
            discriminator_updater = GSDR(state_dim=state_dim, action_dim=action_dim, args=args, policy_updater=policy_updater)

        # discriminator_updater.print_obs_stats()
        if args.norm_obs:
            discriminator_updater.normalize_expert_data(env.norm_mean, env.norm_std)

        # pretrain pi for robosuite env. 
        if args.env_robosuite :
            discriminator_updater.behavior_cloning(policy_net=policy_updater.policy_net, learning_rate=args.learning_rate_pv, bc_step=args.bc_step) # pretrain pi 
        elif args.il_method == "vild":  # pretrain only q_psi
            discriminator_updater.behavior_cloning(policy_net=None, learning_rate=args.learning_rate_pv, bc_step=args.bc_step) 
        elif args.il_method in ["infogail", "infogsd", "infogsdr"]:  # pretrain only policy
            discriminator_updater.behavior_cloning(policy_net=policy_updater.policy_net, learning_rate=args.learning_rate_pv, bc_step=args.bc_step)

        # add bc augmentation
        if "bc" in args.rl_method:
            policy_updater.add_demos(discriminator_updater, args)

    """ Set method and hyper parameter in file name"""
    if method_type == "RL":
        method_name = args.rl_method.upper()
        hypers = rl_hypers_parser(args)    
    else:
        method_name = args.il_method.upper() + "_" + args.rl_method.upper()
        hypers = rl_hypers_parser(args) + "_" + irl_hypers_parser(args)         
        
        if args.il_method == "vild" and args.vild_loss_type.lower() != "linear":
            method_name += "_" + args.vild_loss_type.upper()   
        
        if args.il_method == "infogail" and args.info_loss_type.lower() != "bce":
            method_name += "_" + args.info_loss_type.upper()

    if method_type == "RL":
        exp_name = "%s-%s_s%d" % (method_name, hypers, args.seed)
    elif method_type == "IL":
        exp_name = "%s-%s-%s_s%d" % (discriminator_updater.traj_name, method_name, hypers, args.seed)

    """ Set path for result and model files """
    logsdir = args.logsdir
    assert os.path.exists(logsdir)

    dt_str = datetime.now().strftime('%m.%d_%H.%M.%S')
    algodir = "%s/results_%s/%s/%s" % (logsdir, method_type, env_name, method_name)
    # scan algodir and predix exp_name with idx for easier id from list
    pathlib.Path(algodir).mkdir(parents=True, exist_ok=True)
    itemid = len(os.listdir(algodir))
    parentdir = "%s/%d_%s/%s" % (algodir, itemid, exp_name, dt_str)
    result_path = "%s/results" % (parentdir)
    model_path = "%s/models/ckpt" % (parentdir)
    imglogpath = "%s/img" % (parentdir)
    pathlib.Path(parentdir).mkdir(parents=True)
    pathlib.Path(os.path.dirname(model_path)).mkdir(parents=True)
    pathlib.Path(imglogpath).mkdir(parents=True)
    if logger_fn is None:
        wandb.init(project='vild-solo', config=args, dir=parentdir)
        logger_fn = wandb.log
    else:
        print('***WARNING***: WandB logging disabled')

    def logger_add_scalar(prefix, key, val, gstep):
        logger_fn({f'{prefix}/{key}': val}, gstep)
    def logger_add_histogram(prefix, key, vals, gstep):
        logger_fn({f'{prefix}/{key}': wandb.Histogram(vals)}, gstep)
    def logger_add_image(key, img, gstep):
        if args.tb_log_img == 1:
            logger_fn({key: wandb.Image(img.transpose(1, 2, 0))}, gstep)
        else:
            imageio.imwrite(os.path.join(imglogpath, f"{key.split('/')[-1]}_{gstep}.png"), img.transpose(1, 2, 0))
    def logger_add_elems_from_dict(prefix, dtvalues, gstep):
        if dtvalues is None or not isinstance(dtvalues, dict):
            return
        for k, v in dtvalues.items():
            if isinstance(v, (float, int)):
                logger_add_scalar(prefix, k, v, gstep)
            elif isinstance(v, np.ndarray):
                logger_add_histogram(prefix, k, v, gstep)

    # result_path = "%s/results_%s/%s/%s/%s-%s" % (logsdir, method_type, method_name, env_name, env_name, exp_name)
    # model_path = "%s/results_%s/%s_models/%s/%s-%s" % (logsdir, method_type, method_name, env_name, env_name, exp_name) 
    # pathlib.Path("%s/results_%s/%s/%s" % (logsdir, method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 
    # # if platform.system() != "Windows":
    # pathlib.Path("%s/results_%s/%s_models/%s" % (logsdir, method_type, method_name, env_name)).mkdir(parents=True, exist_ok=True) 

    print("Running %s" % (colored(method_name, p_color)))
    print("%s result will be saved at %s" % (colored(method_name, p_color), colored(result_path, p_color)))

    """ Function to update the parameters of value and policy networks"""
    def update_params_g(batch):
        states = torch.FloatTensor(np.stack(batch.state)).to(device)
        next_states = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        masks = torch.FloatTensor(np.stack(batch.mask)).to(device).unsqueeze(-1)

        actions = torch.LongTensor(np.stack(batch.action)).to(device) if is_disc_action else torch.FloatTensor(np.stack(batch.action)).to(device) 

        if method_type == "RL":
            rewards = torch.FloatTensor(np.stack(batch.reward)).to(device).unsqueeze(-1)
            stats_rl = policy_updater.update_policy(states, actions.to(device), next_states, rewards*args.r_scale, masks)
        elif method_type == "IL":
            nonlocal d_rewards 
            nonlocal _p_rewards 
            # Append one-hot vector of context to state.
            if info_method:
                if not args.encode_cont:
                    latent_codes = torch.LongTensor(np.stack(batch.latent_code)).to(device).view(-1,1)    # [batch_size, 1] 
                else:
                    latent_codes = torch.FloatTensor(np.stack(batch.latent_code)).to(device).view(-1,encode_dim)
                _p_rewards = discriminator_updater.compute_posterior_reward(states, actions, latent_codes, next_states).detach().data
                d_rewards = _p_rewards.clone()

                if not args.encode_cont:
                    latent_codes_onehot = torch.FloatTensor(states.size(0), encode_dim).to(device)
                    latent_codes_onehot.zero_()
                    latent_codes_onehot.scatter_(1, latent_codes, 1)  #should have size [batch_size, num_worker]
                else:
                    latent_codes_onehot = latent_codes

                states_coded = torch.cat((states, latent_codes_onehot), 1) 
                next_states_coded = torch.cat((next_states, latent_codes_onehot), 1)  

                d_rewards += discriminator_updater.compute_reward(states, actions, next_states, codes=latent_codes_onehot).detach().data

                stats_rl = policy_updater.update_policy(states_coded, actions, next_states_coded, d_rewards*args.r_scale, masks)
            else:
                d_rewards = discriminator_updater.compute_reward(states, actions, next_states).detach().data
                stats_rl = policy_updater.update_policy(states, actions, next_states, d_rewards*args.r_scale, masks)
        return stats_rl

    """ Storage and counters """
    memory = Memory(capacity=1000000)   # Memory buffer with 1 million max size.
    if args.exp_d:
        experience = Memory(capacity=int(args.exp_d*args.big_batch_size))   # Buffer for disc sampling
    step, i_iter, tt_g, tt_d, perform_test = 0, 0, 0, 0, 0
    d_rewards = torch.FloatTensor(1).fill_(0)   ## placeholder
    _p_rewards = torch.FloatTensor(1).fill_(0)   ## placeholder
    log_interval = args.max_step // 1000     # 1000 lines in the text files
    if args.env_robosuite:
        log_interval = args.max_step // 500 # reduce to 500 lines to save experiment time
    save_model_interval = (log_interval * args.save_freq) # * (platform.system() != "Windows")  # do not save model ?
    print("Max steps: %s, Log interval: %s steps, Model interval: %s steps" % \
         (colored(args.max_step, p_color), colored(log_interval, p_color), colored(save_model_interval, p_color)))

    """ Reset seed again """  
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    """ Agent for testing in a separated environemnt """
    agent_test = Agent(env_test, render=args.render, t_max=args.t_max, test_cpu=test_cpu)
    agents_gentest = None
    if args.env_name_gentest is not None:
        org_env_name = args.env_name
        agents_gentest = []
        for env_name_gentest in args.env_name_gentest:
            args.env_name = env_name_gentest
            env_gentest = make_env(args)[0]
            agent_gentest = Agent(env_gentest, render=args.render, t_max=args.t_max, test_cpu=test_cpu)
            agents_gentest.append(agent_gentest)
        args.env_name = org_env_name
    if args.env_bullet: 
        log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)

    latent_code = None ## only for infogail 
    state = env.reset()
    done = 1 
    stats_disc = None
    stats_rl = None
    """ The actual learning loop"""
    for total_step in range(0, args.max_step + 1):

        """ Save the learned policy model """
        if save_model_interval > 0 and total_step % save_model_interval == 0: 
            policy_updater.save_model("%s_policy_T%d.pt" % (model_path, total_step))
            if args.il_method is not None:
                discriminator_updater.save_model("%s_discr_T%d.pt" % (model_path, total_step))

        """ Test the policy before update """
        if total_step % log_interval == 0:
            perform_test = 1
         
        """ Test learned policy """
        if perform_test:
            if not info_method:
                if not args.env_bullet:
                    log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                    perform_test = 0
                elif done: # Because env and env_test are the same object for pybullet. 
                    log_test = agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10)
                    perform_test = 0
            else:
                log_test = []
                if not args.encode_cont:
                    for i_k in range(0, encode_dim):
                        latent_code_test = torch.LongTensor(size=(1,1)).fill_(i_k)
                        latent_code_onehot_test = torch.FloatTensor(1, encode_dim)
                        latent_code_onehot_test.zero_()
                        latent_code_onehot_test.scatter_(1, latent_code_test, 1)
                        log_test += [agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=10, latent_code_onehot=latent_code_onehot_test.squeeze() )] # use 1 instead of 10 to save time?
                else:
                    for i_k in range(0, 25):
                        latent_code_test = discriminator_updater.sample_code().squeeze(dim=0)
                        log_test += [agent_test.collect_samples_test(policy=policy_updater, max_num_episodes=1, latent_code_onehot=latent_code_test )]
                perform_test = 0

        latent_archive_id = -1
        if info_method and latent_code is None:
            latent_code = discriminator_updater.sample_code(train=True)
            latent_archive_id = discriminator_updater.sampled_archive_id
            if not args.encode_cont:
                latent_code_onehot = torch.FloatTensor(1, encode_dim)
                latent_code_onehot.zero_()
                latent_code_onehot.scatter_(1, latent_code, 1)
                latent_code_onehot = latent_code_onehot.squeeze()  #should have size [encode_dim]
            else:
                latent_code_onehot = latent_code.squeeze(dim=0)
            latent_code = latent_code.detach().numpy()

        state_var = torch.FloatTensor(state)
        if latent_code is not None:
            state_var = torch.cat((state_var, latent_code_onehot), 0)  

        """ take env step """
        if total_step <= args.random_action and update_type == "off_policy":    # collect random actions first for off policy methods
            action = env.action_space.sample()
        else:
            action = policy_updater.sample_action(state_var.to(device).unsqueeze(0)).to(device_cpu).detach().numpy()

        if args.il_method == "vild":    # Add noise from Sigma_k to action (noise_t = sqrt(Sigma_k)) 
            action_u = action + args.noise_t * np.random.normal( np.zeros(action.shape), np.ones(action.shape) )
            next_state, reward, done, _ = env.step(action_u)
        else:
            next_state, reward, done, _ = env.step(action)

        if step + 1 == args.t_max:
            done = 1
        memory.push(state, action, int(not done), next_state, reward, latent_code, latent_archive_id)
        if args.exp_d:
            experience.push(state, action, int(not done), next_state, reward, latent_code, latent_archive_id)
        state = next_state
        step = step + 1        

        """ reset env """
        if done :  # reset
            state = env.reset()
            step = 0
            latent_code = None 
                        
        """ Curriculum related updates """
        policy_updater.update_curriculum(total_step/args.max_step)
        if args.il_method is not None:
            discriminator_updater.update_curriculum(total_step/args.max_step)

        """ Update policy """
        if update_type == "on_policy":
            if memory.size() >= args.big_batch_size and done :
                batch = memory.sample()
                if method_type == "IL":
                    batch_d = experience.sample(args.big_batch_size) if args.exp_d else batch
                    for i_d in range(0, args.d_step):
                        index = discriminator_updater.index_sampler()   # should be inside update_discriminator for cleaner code...
                        t0_d = time.time()
                        stats_disc = discriminator_updater.update_discriminator(batch=batch_d, index=index, total_step=total_step)
                        tt_d += time.time() - t0_d

                t0_g = time.time()
                stats_rl = update_params_g(batch=batch)
                tt_g += time.time() - t0_g
                memory.reset()

        elif update_type == "off_policy":
            if total_step >= args.big_batch_size:       

                if method_type == "IL":
                    index = discriminator_updater.index_sampler()
                    batch = memory.sample(args.mini_batch_size)    
                    t0_d = time.time()
                    stats_disc = discriminator_updater.update_discriminator(batch=batch, index=index, total_step=total_step)
                    tt_d += time.time() - t0_d  
                elif method_type == "RL":
                    batch = memory.sample(args.mini_batch_size)    
                    
                t0_g = time.time()
                stats_rl = update_params_g(batch=batch)  
                tt_g += time.time() - t0_g
                       
        """ Print out result to stdout and save it to a text file for plotting """
        if total_step % log_interval == 0:
            tb_log_step = int(total_step // log_interval)
            logger_add_elems_from_dict('disc', stats_disc, tb_log_step)
            logger_add_elems_from_dict('rl', stats_rl, tb_log_step)

            result_text = t_format("Step %7d " % (total_step), 0) 
            if method_type == "RL":
                result_text += t_format("(g%2.2f)s" % (tt_g), 1)  
            elif method_type == "IL":
                c_reward_list = d_rewards.to(device_cpu).detach().numpy()
                result_text += t_format("(g%2.1f+d%2.1f)s" % (tt_g, tt_d), 1) 
                result_text += " | [D] " + t_format("min: %.2f" % np.amin(c_reward_list), 0.5) + t_format(" max: %.2f" % np.amax(c_reward_list), 0.5)
                logger_add_scalar('misc', 'd_rewards/min', np.amin(c_reward_list), tb_log_step)
                logger_add_scalar('misc', 'd_rewards/max', np.amax(c_reward_list), tb_log_step)
                if info_method:
                    p_reward_list = _p_rewards.to(device_cpu).detach().numpy()
                    logger_add_scalar('misc', 'p_rewards/min', np.amin(p_reward_list), tb_log_step)
                    logger_add_scalar('misc', 'p_rewards/max', np.amax(p_reward_list), tb_log_step)

            result_text += " | [R_te] "
            if not info_method:
                result_text += t_format("min: %.2f" % log_test['min_reward'], 1) + t_format("max: %.2f" % log_test['max_reward'], 1) \
                    + t_format("Avg: %.2f (%.2f)" % (log_test['avg_reward'], log_test['std_reward']), 2)
                logger_add_scalar('misc', 'e_rewards/min', log_test['min_reward'], tb_log_step)
                logger_add_scalar('misc', 'e_rewards/max', log_test['max_reward'], tb_log_step)
                logger_add_scalar('misc', 'e_rewards/mean', log_test['avg_reward'], tb_log_step)
                logger_add_scalar('misc', 'e_rewards/std', log_test['std_reward'], tb_log_step)
            else:        
                result_text += "Avg " 
                if not args.encode_cont:
                    for i_k in range(0, encode_dim):
                        result_text += t_format("%d: %.2f (%.2f)" % (i_k, log_test[i_k]['avg_reward'], log_test[i_k]['std_reward']), 2)
                else:
                    rewards = [ilog['avg_reward'] for ilog in log_test]
                    ravg = np.mean(rewards)
                    rstd = np.std(rewards)
                    result_text += t_format("%.2f (%.2f)" % (ravg, rstd), 2)
                    logger_add_scalar('misc', 'e_rewards/mean', ravg, tb_log_step)
                    logger_add_scalar('misc', 'e_rewards/std', rstd, tb_log_step)

                    lastrewards = [ilog['avg_lastreward'] for ilog in log_test]
                    lravg = np.mean(lastrewards)
                    lrstd = np.std(lastrewards)
                    result_text += t_format("%.2f (%.2f)" % (lravg, lrstd), 2)
                    logger_add_scalar('misc', 'e_lastrewards/mean', lravg, tb_log_step)
                    logger_add_scalar('misc', 'e_lastrewards/std', lrstd, tb_log_step)

            if (args.rl_method == "sac"):
                result_text += ("| ent %0.3f" % (policy_updater.entropy_coef))
                logger_add_scalar('misc', 'sac/ent', policy_updater.entropy_coef, tb_log_step)

            if args.il_method == "vild":
                ## check estimated worker noise
                estimated_worker_noise = discriminator_updater.worker_net.get_worker_cov().to(device_cpu).detach().numpy().squeeze()
                if action_dim > 1:
                    estimated_worker_noise = estimated_worker_noise.mean(axis=0)  #average across action dim
                result_text += " | w_noise: %s" % (np.array2string(estimated_worker_noise, formatter={'float_kind':lambda x: "%.5f" % x}).replace('\n', '') )
                    
            tt_g = 0
            tt_d = 0

            print(result_text)
            with open(result_path + ".txt", 'a') as f:
                print(result_text, file=f) 
            
            if tb_log_step % args.plot_freq == 0 and not info_method:
                rollouts = []
                for i_k in range(0, 10):
                    rollouts += agent_test.collect_rollouts_test(policy=policy_updater, max_num_episodes=1, latent_code_onehot=None)

                if 'pend' in env_name.lower():
                    metrics_keys = rollouts[0]['infos'][0].keys()
                    metrics_vals = {mk: [] for mk in metrics_keys}
                    for r_i in rollouts:
                        for mk in metrics_keys:
                            metrics_vals[mk].append(np.mean([info_dict[mk] for info_dict in r_i['infos']]))
                    for mk, mvs in metrics_vals.items():
                        logger_add_scalar(f'misc/pend/{mk}', 'mean', np.mean(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'std', np.std(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'max', np.max(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'min', np.min(mvs), tb_log_step)

                if 'hc' in env_name.lower():
                    metrics_keys = rollouts[0]['infos'][0].keys()
                    metrics_vals = {mk: [] for mk in metrics_keys}
                    for r_i in rollouts:
                        for mk in metrics_keys:
                            metrics_vals[mk].append(np.mean([info_dict[mk] for info_dict in r_i['infos']]))
                    for mk, mvs in metrics_vals.items():
                        logger_add_scalar(f'misc/hc/{mk}', 'mean', np.mean(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'std', np.std(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'max', np.max(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'min', np.min(mvs), tb_log_step)

            if tb_log_step % args.plot_freq == 0 and info_method:
                # if agents_gentest is not None:
                if agents_gentest is not None and tb_log_step > 0:
                    # extra testing for generalization
                    for i, agent_gentest in enumerate(agents_gentest):
                        log_gentest = []
                        for i_k in range(0, 25):
                            latent_code_test = discriminator_updater.sample_code().squeeze(dim=0)
                            log_gentest += [agent_gentest.collect_samples_test(policy=policy_updater, max_num_episodes=1, latent_code_onehot=latent_code_test)]
                        l_rets = [log['avg_reward'] for log in log_gentest]
                        logger_add_scalar(f'misc/gen_test{i}/e_rewards', 'mean', np.mean(l_rets), tb_log_step)
                        logger_add_scalar(f'misc/gen_test{i}/e_rewards', 'std', np.std(l_rets), tb_log_step)
                        logger_add_scalar(f'misc/gen_test{i}/e_rewards', 'max', np.max(l_rets), tb_log_step)
                        logger_add_scalar(f'misc/gen_test{i}/e_rewards', 'min', np.min(l_rets), tb_log_step)

                rolloutsets = []
                # for i_k in range(0, 50 if tb_log_step > 0 else 1):
                # for i_k in range(0, 100):
                for i_k in range(0, 40 if tb_log_step > 0 else 1):
                    latent_code_test = discriminator_updater.sample_code().squeeze(dim=0)
                    # max_num_episodes = 5
                    max_num_episodes = 1 if True in [envtok in env_name.lower() for envtok in ['fetchpickplace', 'pend']] else 5
                    rolloutsets.append(agent_test.collect_rollouts_test(policy=policy_updater, max_num_episodes=max_num_episodes, latent_code_onehot=latent_code_test))
                # all dicts copied by reference
                rollouts = [r for rollouts_per_latent in rolloutsets for r in rollouts_per_latent]

                # scatter plot zs from archive
                if hasattr(discriminator_updater, 'get_archive_numpy'):
                    codes_archive = discriminator_updater.get_archive_numpy()
                    archive_fig = agent_test.build_code_scatter_fig(codes_archive)
                    logger_add_image('misc/archive', archive_fig, tb_log_step)

                r_obss = [r['observations'] for r in rollouts]
                logger_add_scalar('misc/normed_obs', 'mean', np.mean(r_obss), tb_log_step)
                logger_add_scalar('misc/normed_obs', 'std', np.std(r_obss), tb_log_step)
                logger_add_scalar('misc/normed_obs', 'max', np.max(r_obss), tb_log_step)
                logger_add_scalar('misc/normed_obs', 'min', np.min(r_obss), tb_log_step)

                # unnormalize obs for dataset metrics + figs
                if args.norm_obs:
                    for i in range(len(rollouts)):
                        rollouts[i]['observations'] = env.unnorm_obs(rollouts[i]['observations'])

                dset_stats = agent_test.calc_dataset_metrics(discriminator_updater, policy_updater, rolloutsets, budget=10)
                logger_add_elems_from_dict('misc/dataset', dset_stats, tb_log_step)
                logger_add_scalar('misc', 'diversity', agent_test.calc_diversity(rolloutsets), tb_log_step)

                # created here as heatmap/rep fn needs this
                norm_fn = env.observation if args.norm_obs else lambda x: x
                skills_fig = None
                if 'maze' in env_name.lower():
                    if 'gail' in args.il_method:
                        skills_fig = agent_test.build_skills_traces_fig(rollouts, dims=[0, 1])
                        logger_add_image('misc/maze/skills_fig', skills_fig, tb_log_step)
                        disc_fig = agent_test.build_disc_heat_fig(discriminator_updater, norm_fn=norm_fn)
                        logger_add_image('misc/maze/disc_fig', disc_fig, tb_log_step)
                    if 'gsd' in args.il_method:
                        # skills + embspace + embtraces
                        skills_fig = agent_test.build_skills_space_traces_fig(
                            rollouts, discriminator_updater, norm_fn=norm_fn)
                        logger_add_image('misc/maze/skills_fig', skills_fig, tb_log_step)
                        # discriminator
                        if not getattr(discriminator_updater, 'cond_rew', False):
                            disc_fig = agent_test.build_disc_heat_fig(discriminator_updater, norm_fn=norm_fn)
                            logger_add_image('misc/maze/disc_fig', disc_fig, tb_log_step)
                        elif args.encode_dim == 2:
                            if args.il_method == 'infogsd':
                                disc_fig = agent_test.build_cdisc_heat_fig(discriminator_updater, norm_fn=norm_fn, heat_max=args.clip_discriminator)
                            if args.il_method == 'infogsdr':
                                disc_fig = agent_test.build_gdisc_heat_fig(discriminator_updater, norm_fn=norm_fn, heat_max=args.clip_discriminator)
                            logger_add_image('misc/maze/disc_fig', disc_fig, tb_log_step)
                if 'pend' in env_name.lower():
                    skills_fig = agent_test.build_skills_hist_fig(rollouts, dim=0)
                    logger_add_image('misc/pend/skills_hist', skills_fig, tb_log_step)
                    metrics_vals = {}
                    for mk in rolloutsets[0][0]['infos'][0].keys():
                        metrics_vals[mk+'_mean'] = []
                        metrics_vals[mk+'_std'] = []
                        for r_set in rolloutsets:
                            r_i_mvs = []
                            for r_i in r_set:
                                r_i_mvs.append(np.mean([info_dict[mk] for info_dict in r_i['infos']]))
                            metrics_vals[mk+'_mean'].append(np.mean(r_i_mvs))
                            metrics_vals[mk+'_std'].append(np.std(r_i_mvs))
                    for mk, mvs in metrics_vals.items():
                        logger_add_scalar(f'misc/pend/{mk}', 'mean', np.mean(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'std', np.std(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'max', np.max(mvs), tb_log_step)
                        logger_add_scalar(f'misc/pend/{mk}', 'min', np.min(mvs), tb_log_step)
                    logger_add_histogram('misc/pend', 'ramp_pos_mean', np.array(metrics_vals['ramp_pos_mean']), tb_log_step)
                    logger_add_histogram('misc/pend', 'ramp_pos_std', np.array(metrics_vals['ramp_pos_std']), tb_log_step)
                    if args.v_data <= 1:
                        ramp_pos_targets = [-0.4, -0.2, 0.0, 0.2, 0.4]
                    elif args.v_data == 2:
                        ramp_pos_targets = [-0.721, -0.374, 0.0, 0.407, 0.741]
                    recv_stats = agent_test.calc_recovery_metrics(
                        metrics_vals,
                        'ramp_pos_mean',
                        targets=ramp_pos_targets,
                        budget=10,
                    )
                    logger_add_elems_from_dict('misc/pend/recovery', recv_stats, tb_log_step)
                if 'hopper' in env_name.lower():
                    skills_fig = agent_test.build_skills_hist_fig(rollouts, dim=3)
                    logger_add_image('misc/hopper/skills_hist', skills_fig, tb_log_step)
                if 'hc' in env_name.lower():
                    metrics_vals = {}
                    for mk in rolloutsets[0][0]['infos'][0].keys():
                        metrics_vals[mk+'_mean'] = []
                        metrics_vals[mk+'_std'] = []
                        for r_set in rolloutsets:
                            r_i_mvs = []
                            for r_i in r_set:
                                r_i_mvs.append(np.mean([info_dict[mk] for info_dict in r_i['infos']]))
                            metrics_vals[mk+'_mean'].append(np.mean(r_i_mvs))
                            metrics_vals[mk+'_std'].append(np.std(r_i_mvs))
                    for mk, mvs in metrics_vals.items():
                        logger_add_scalar(f'misc/hc/{mk}', 'mean', np.mean(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'std', np.std(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'max', np.max(mvs), tb_log_step)
                        logger_add_scalar(f'misc/hc/{mk}', 'min', np.min(mvs), tb_log_step)
                    logger_add_histogram('misc/hc', 'current_vel_mean', np.array(metrics_vals['current_vel_mean']), tb_log_step)
                    logger_add_histogram('misc/hc', 'current_vel_std', np.array(metrics_vals['current_vel_std']), tb_log_step)
                    recv_stats = agent_test.calc_recovery_metrics(
                        metrics_vals,
                        'current_vel_mean',
                        targets=[0.97, 1.78, 2.88, 3.65, 4.87],
                        budget=10,
                    )
                    logger_add_elems_from_dict('misc/hc/recovery', recv_stats, tb_log_step)
                if 'fetch' in env_name.lower():
                    if 'pushsingle' in env_name.lower():
                        ftop = agent_test.build_skills_traces_fig(rollouts, dims=[1, 2], bounds=((0.0, 1.5), (0.4, 0.9)))
                        fbot1 = agent_test.build_skills_traces_fig(rollouts, dims=[4, 10], bounds=((0.0, 1.5), (-0.1, 0.1)))
                        fbot2 = agent_test.build_skills_traces_fig(rollouts, dims=[3, 4], bounds=((1.0, 1.6), (0.0, 1.5)))
                        fbot = np.concatenate([fbot1[:, :, :500], fbot2[:, :, :500]], axis=2)
                        skills_fig = np.concatenate([ftop, fbot], axis=1)
                        logger_add_image('misc/fetch/skills_alltraces', skills_fig, tb_log_step)
                    if 'pushdouble' in env_name.lower():
                        skills_fig = agent_test.build_skills_traces_fig(rollouts, dims=[1, 2], bounds=((0.0, 1.5), (0.4, 0.9)))
                        logger_add_image('misc/fetch/skills_gpyztraces', skills_fig, tb_log_step)
                        skills_fig = agent_test.build_skills_traces_fig(rollouts, dims=[4, 7], bounds=((0.6, 0.9), (0.6, 0.9)))
                        logger_add_image('misc/fetch/skills_o12ytraces', skills_fig, tb_log_step)
                    if 'pickplace' in env_name.lower():
                        ftop = agent_test.build_skills_traces_fig(rollouts, dims=[0, 2], bounds=((1.0, 1.6), (0.2, 0.8)))
                        fbot = agent_test.build_skills_traces_fig(rollouts, dims=[3, 4], bounds=((1.0, 1.6), (0.5, 1.5)))
                        halfwidth = fbot.shape[2] // 2
                        skills_fig = np.concatenate([ftop[:, :, :halfwidth], fbot[:, :, :halfwidth]], axis=1)
                        logger_add_image('misc/fetch/skills_alltraces', skills_fig, tb_log_step)
                        metrics_vals = {}
                        for mk in rolloutsets[0][0]['infos'][0].keys():
                            metrics_vals[mk+'_mean'] = []
                            metrics_vals[mk+'_std'] = []
                            for r_set in rolloutsets:
                                r_i_mvs = []
                                for r_i in r_set:
                                    # val_ro = np.mean([info_dict[mk] for info_dict in r_i['infos']])
                                    val_ro = r_i['infos'][-1][mk]
                                    r_i_mvs.append(val_ro)
                                r_i_mvs = np.array(r_i_mvs)
                                metrics_vals[mk+'_mean'].append(np.mean(r_i_mvs))
                                metrics_vals[mk+'_std'].append(np.std(r_i_mvs))
                        for mk, mvs in metrics_vals.items():
                            logger_add_scalar(f'misc/fetch/{mk}', 'mean', np.mean(mvs), tb_log_step)
                            logger_add_scalar(f'misc/fetch/{mk}', 'std', np.std(mvs), tb_log_step)
                            logger_add_scalar(f'misc/fetch/{mk}', 'max', np.max(mvs), tb_log_step)
                            logger_add_scalar(f'misc/fetch/{mk}', 'min', np.min(mvs), tb_log_step)
                        logger_add_histogram('misc/fetch', 'obj_pos_y_mean', np.array(metrics_vals['obj_pos_y_mean']), tb_log_step)
                        objy_targets = np.linspace(-0.8, 0.8, 5)*0.2 + 0.75
                        if 'wide' in env_name.lower():
                            objy_targets = np.linspace(-0.8, 0.8, 5)*0.4 + 0.75
                        recv_stats = agent_test.calc_recovery_metrics(
                            metrics_vals,
                            'obj_pos_y_mean',
                            targets=objy_targets,
                            budget=10,
                        )
                        logger_add_elems_from_dict('misc/fetch/recovery', recv_stats, tb_log_step)

if __name__ == "__main__":
    args = args_parser()
    args.tb_log_img = 1
    logger_fn = None
    logger_fn = lambda x, y: None
    main(args, logger_fn)
