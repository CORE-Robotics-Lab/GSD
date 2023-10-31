from my_utils import *
import matplotlib; matplotlib.use('agg')
from matplotlib import pyplot as plt
import itertools
from torch.utils.tensorboard._utils import figure_to_image

def chunks(arr, nchunks):
    chunksize = int(np.ceil(len(arr) / nchunks))
    arrlist = [arr[i*chunksize:(i+1)*chunksize] for i in range(nchunks)]
    return arrlist

class Agent:

    def __init__(self, env, render=0, clip=False, t_max=1000, test_cpu=True):
        self.env = env
        self.render = render
        self.test_cpu = test_cpu
        self.t_max = t_max      
        self.is_disc_action = len(env.action_space.shape) == 0

    def collect_samples_test(self, policy, max_num_episodes, latent_code_onehot=None ):
        log = dict()
        min_reward = 1e6
        max_reward = -1e6
        total_reward_list = []
        last_reward_list = []

        if self.test_cpu:
            policy.policy_to_device(device_cpu)
            device_x = device_cpu 
        else:
            device_x = device 

        for _ in range(0, max_num_episodes):
            reward_episode = 0
            state = self.env.reset()

            step = 0
            while True: # run an episode

                state_var = torch.FloatTensor(state)
                if latent_code_onehot is not None:
                    state_var = torch.cat((state_var, latent_code_onehot), 0)  

                action = policy.greedy_action(state_var.to(device_x).unsqueeze(0)).to(device_cpu).detach().numpy()

                next_state, reward, done, _ = self.env.step(action)    

                if step + 1 == self.t_max:
                    done = 1    

                reward_episode += reward   

                if self.render:
                    self.env.render(mode="human")
                    time.sleep(0.001)
                    
                if done:
                    break
                                 
                state = next_state
                step = step + 1    

            # log stats
            total_reward_list += [reward_episode]
            min_reward = min(min_reward, reward_episode)
            max_reward = max(max_reward, reward_episode)
            last_reward_list += [reward]

        if self.test_cpu:
            policy.policy_to_device(device)

        log['avg_reward'] = np.mean(np.array(total_reward_list))   
        log['std_reward'] = np.std(np.array(total_reward_list)) / np.sqrt(max_num_episodes) 
        log['max_reward'] = max_reward
        log['min_reward'] = min_reward
        log['avg_lastreward'] = np.mean(np.array(last_reward_list))
        return log

    def collect_rollouts_test(self, policy, max_num_episodes, latent_code_onehot=None):
        rollouts = []

        if self.test_cpu:
            policy.policy_to_device(device_cpu)
            device_x = device_cpu 
        else:
            device_x = device 

        for _ in range(0, max_num_episodes):
            state = self.env.reset()

            rollout = {
                'observations': [state.copy()],
                'actions': [],
                'code': latent_code_onehot.data.numpy() if latent_code_onehot is not None else None,
                'infos': [],
            }
            step = 0
            while True: # run an episode

                state_var = torch.FloatTensor(state)
                if latent_code_onehot is not None:
                    state_var = torch.cat((state_var, latent_code_onehot), 0)  

                action = policy.greedy_action(state_var.to(device_x).unsqueeze(0)).to(device_cpu).detach().numpy()

                next_state, reward, done, info = self.env.step(action)    

                if step + 1 == self.t_max:
                    done = 1    

                rollout['observations'].append(next_state.copy())
                rollout['actions'].append(action.copy())
                rollout['infos'].append(info)

                if self.render:
                    self.env.render(mode="human")
                    time.sleep(0.001)
                    
                if done:
                    break
                                 
                state = next_state
                step = step + 1    

            # log stats
            rollouts.append({k: np.array(v) for k, v in rollout.items()})

        if self.test_cpu:
            policy.policy_to_device(device)

        return rollouts

    def calc_recovery_metrics(self, metrics_all, mkey, targets, budget=10):

        if len(metrics_all[mkey]) < budget:
            return {}

        mvals_all = np.array(metrics_all[mkey])
        mvals_chunks = chunks(mvals_all, len(mvals_all) // budget)

        mvalerrors = []
        for it, tgtvel in enumerate(targets):

            errors_pertgt = []
            # for code_chunk, vel_chunk in zip(codes_chunks, mvals_chunks):
            for vel_chunk in mvals_chunks:
                dist_chunk = np.abs(vel_chunk - tgtvel)
                didxs_chunk = np.argsort(dist_chunk)
                # tgtcodes = code_chunk[didxs_chunk[:1]]
                # tgtcodes = [[-1.1880176, -0.5497462 ]]

                # sample 5 traj with latent
                # _, veltestrolls = run_rollouts(args, env, policy_updater, torch.FloatTensor(tgtcodes[0]), 5)
                # testvel_chunk = np.array([np.mean([ifdict['current_vel'] for ifdict in ri['infos']]) for ri in veltestrolls])
                # use same traj
                testvel_chunk = vel_chunk[didxs_chunk[:1]]

                # print(tgtvel, tgtcodes[0], testvel_chunk)
                tgtvelerrors = np.abs(testvel_chunk - tgtvel)
                errors_pertgt.append(tgtvelerrors)
            errors_pertgt = np.concatenate(errors_pertgt, axis=0)

            # print(f'VL-{it+1}', np.mean(errors_pertgt), np.std(errors_pertgt))
            mvalerrors.append(np.mean(errors_pertgt))

        return {
            f'{mkey}/mae/mean': np.mean(mvalerrors),
            f'{mkey}/mae/std': np.std(mvalerrors),
        }

    def calc_dataset_metrics(self, du, pi, rolloutsets_all, budget=10):

        if len(rolloutsets_all) < budget:
            return {}

        kl_s_train = []
        kl_sa_train = []
        kl_s_val = []
        kl_sa_val = []
        kl_s_test = []
        kl_sa_test = []

        for rolloutsets in chunks(rolloutsets_all, len(rolloutsets_all)//budget):

            def calc_kl(exp_s, exp_a, include_action):
                lps_dset = []
                for states_d, actions_d in zip(exp_s, exp_a):
                    if not include_action:
                        vec_d = states_d
                    else:
                        vec_d = np.concatenate([states_d, actions_d], axis=1)
                    mvn_d = calc_mvn_params(vec_d)

                    lps = []
                    for rollouts in rolloutsets:
                        vec_r = []
                        for r in rollouts:
                            if not include_action:
                                vec_ri = r['observations']
                            else:
                                vec_ri = np.concatenate([r['observations'][:-1], r['actions']], axis=1)
                            vec_r.append(vec_ri)
                        vec_r = np.concatenate(vec_r, axis=0)
                        mvn_r = calc_mvn_params(vec_r)

                        _lp = kl_mvn(mvn_d, mvn_r)
                        lps.append(_lp)
                    lps_dset.append(np.min(lps))
                return lps_dset

            # these are unnormalized
            kl_s_train += calc_kl(du.expert_states_train, du.expert_actions_train, include_action=False)
            kl_sa_train += calc_kl(du.expert_states_train, du.expert_actions_train, include_action=True)
            kl_s_val += calc_kl(du.expert_states_val, du.expert_actions_val, include_action=False)
            kl_sa_val += calc_kl(du.expert_states_val, du.expert_actions_val, include_action=True)
            kl_s_test += calc_kl(du.expert_states_test, du.expert_actions_test, include_action=False)
            kl_sa_test += calc_kl(du.expert_states_test, du.expert_actions_test, include_action=True)
        
        return {
            'kl_s/train': np.mean(kl_s_train),
            'kl_sa/train': np.mean(kl_sa_train),
            'kl_s/val': np.mean(kl_s_val),
            'kl_sa/val': np.mean(kl_sa_val),
            'kl_s/test': np.mean(kl_s_test),
            'kl_sa/test': np.mean(kl_sa_test),
        }

    def calc_diversity(self, rolloutsets):
        if len(rolloutsets) == 1: return 0
        mus = []
        for rollouts in rolloutsets:
            vec_r = []
            for r in rollouts:
                vec_ri = np.concatenate([r['observations'][:-1], r['actions']], axis=1)
                vec_r.append(vec_ri)
            vec_r = np.concatenate(vec_r, axis=0)
            vec_r_mu = np.mean(vec_r, axis=0)
            mus.append(vec_r_mu)
        mus = np.array(mus)

        mus0 = np.expand_dims(mus, axis=0)
        mus1 = np.expand_dims(mus, axis=1)
        dists = np.sum(np.square(mus0 - mus1), axis=2)

        K = mus.shape[0]//2
        dists_k = np.partition(dists, K, axis=1)
        dists_k = dists_k[:, K]
        div = np.sum(np.log(dists_k))
        return div

    def build_code_scatter_fig(self, codes):
        colors = get_code_colors(codes)
        fig = plt.figure(figsize=(5, 5))
        plt.xlim((-3.0, 3.0)); plt.ylim((-3.0, 3.0))
        # offset = 2*np.pi/((4-1)*4*2)
        # thetas = np.linspace(offset, np.pi + offset, num=(4-1)*2, endpoint=False)
        # for tta in thetas:
        #     plt.plot([-5*np.cos(tta), 5*np.cos(tta)], [-5*np.sin(tta), 5*np.sin(tta)],
        #              c='black', alpha=0.2)
        for i, (code, color) in enumerate(zip(codes, colors)):
            if len(code) < 2: code = [code[0], 0]
            plt.scatter(code[0], code[1], s=10, color=color)
            plt.annotate(str(i), code[:2])
        plt.tight_layout()
        return figure_to_image(fig)

    # makes 2D figures of state space traces of specified dims
    def build_skills_traces_fig(self, rollouts, dims=[0, 1], bounds=((-2, 2), (-2, 2))):
        fig = plt.figure(figsize=(10, 5))

        codes = np.array([r['code'] for r in rollouts])
        colors = get_code_colors(codes)

        plt.subplot(1, 2, 1)
        if bounds is not None:
            plt.xlim(bounds[0]); plt.ylim(bounds[1])
        for ir, rollout in enumerate(rollouts):
            obses = rollout['observations']
            plt.plot(obses[:, dims[0]], obses[:, dims[1]], c=colors[ir])
            plt.scatter(obses[:, dims[0]], obses[:, dims[1]], s=3, c='black')

        plt.subplot(1, 2, 2)
        plt.xlim((-3.0, 3.0)); plt.ylim((-3.0, 3.0))
        # offset = 2*np.pi/((4-1)*4*2)
        # thetas = np.linspace(offset, np.pi + offset, num=(4-1)*2, endpoint=False)
        # for tta in thetas:
        #     plt.plot([-5*np.cos(tta), 5*np.cos(tta)], [-5*np.sin(tta), 5*np.sin(tta)],
        #              c='black', alpha=0.2)
        for code, color in zip(codes, colors):
            if len(code) < 2: code = [code[0], 0]
            plt.scatter(code[0], code[1], s=10, color=color)

        plt.tight_layout()
        return figure_to_image(fig)

    # makes histogram figures of mean state vectors of specified dim
    def build_skills_hist_fig(self, rollouts, dim=0):
        fig = plt.figure(figsize=(10, 5))

        codes = np.array([r['code'] for r in rollouts])
        meanstates = np.array([np.mean(r['observations'], axis=0) for r in rollouts])
        colors = get_code_colors(codes)

        cellwidth = 0.5
        codes_cells = (np.around(codes/cellwidth)) * cellwidth
        colors_cells = get_code_colors(codes_cells)
        cells_ids = (codes_cells[:, 0]*2 + 2000*codes_cells[:, 1]).astype(int)

        list_meanstates = []
        list_colors = []
        for cell_id in np.unique(cells_ids):
            idxs = np.where(cell_id == cells_ids)[0]
            list_meanstates.append(meanstates[idxs, dim])
            list_colors.append(colors_cells[idxs[0]])

        plt.subplot(1, 2, 1)
        plt.hist(
            list_meanstates,
            bins=20, 
            density=True,
            rwidth=0.8,
            range=(-1, 1),
            color=list_colors,
            histtype='bar',
            stacked=True,
        )

        plt.subplot(1, 2, 2)
        plt.xlim((-3.0, 3.0)); plt.ylim((-3.0, 3.0))
        for code, color in zip(codes, colors):
            if len(code) < 2: code = [code[0], 0]
            plt.scatter(code[0], code[1], s=10, color=color)

        plt.tight_layout()
        return figure_to_image(fig)

    def build_disc_heat_fig(self, disc, cellhalfwidth=0.2, mazehalfwidth=3.0, norm_fn=None):
        fig = plt.figure(figsize=(5, 4))
        modparam = 0.5 / cellhalfwidth

        # plot learned reward heatmap
        # obs_next
        gpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        gpts = np.array([list(pt) for pt in gpts])
        # obs
        cpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        cpts = np.array([list(pt) for pt in cpts])
        cpts = np.around(cpts * modparam) / modparam
        # acts
        # acts = np.random.normal(size=cpts.shape)
        acts = gpts - cpts
        acts = acts/(np.max(acts)+1e-8)

        def heat_fn(s_np, a_np, s2_np):
            with torch.no_grad():
                assert s_np.ndim == 2 and a_np.ndim == 2 and s2_np.ndim == 2
                s_th = torch.Tensor(norm_fn(s_np))
                a_th = torch.Tensor(a_np)
                s2_th = torch.Tensor(norm_fn(s2_np))
                # r_th = disc.discrim_net.get_reward(s_th, a_th, s2_th)[:, 0]
                r_th = disc.discrim_net.get_reward(s_th, a_th)[:, 0]
            return r_th.data.numpy()
        rews = get_heat_for_xy(cpts, acts, gpts, heat_fn, minimal=False)
        heats = np.array(rews)
        heatmap = plt.pcolormesh(
            trans_arr2grid(gpts[:, 0]),
            trans_arr2grid(gpts[:, 1]),
            trans_arr2grid(heats),
            shading='nearest')
        plt.colorbar(heatmap)

        ngridlines = int(mazehalfwidth / cellhalfwidth)
        xs = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)
        plt.vlines(xs, ymin=-mazehalfwidth, ymax=mazehalfwidth, colors='w', linewidths=1.5)
        ys = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)
        plt.hlines(ys, xmin=-mazehalfwidth, xmax=mazehalfwidth, colors='w', linewidths=1.5)

        plt.tight_layout()
        return figure_to_image(fig)

    def build_cdisc_heat_fig(self, disc, cellhalfwidth=0.2, mazehalfwidth=3.0, heat_max=None, norm_fn=None, gridsize=4):
        fig = plt.figure(figsize=(int(5*3), int(4*3)))

        modparam = 0.5 / cellhalfwidth
        # obs_next
        gpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        gpts = np.array([list(pt) for pt in gpts])
        # obs
        cpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        cpts = np.array([list(pt) for pt in cpts])
        cpts = np.around(cpts * modparam) / modparam
        # acts
        acts = gpts - cpts
        acts = acts/(np.max(acts)+1e-8)

        def get_heat_for_dir(dirv):
            def heat_fn(s_np, a_np, s2_np):
                with torch.no_grad():
                    assert s_np.ndim == 2 and a_np.ndim == 2 and s2_np.ndim == 2
                    s_th = torch.Tensor(norm_fn(s_np))
                    a_th = torch.Tensor(a_np)
                    s2_th = torch.Tensor(norm_fn(s2_np))
                    z_th = torch.tile(torch.Tensor(dirv).unsqueeze(0), (s_th.size(0), 1))
                    sz_th = torch.cat([s_th, z_th], 1)
                    s2z_th = torch.cat([s2_th, z_th], 1)
                    r_th = disc.discrim_net.get_reward(sz_th, a_th, s2z_th, pure_reward=True)[:, 0]
                return r_th.data.numpy()
            rews = get_heat_for_xy(cpts, acts, gpts, heat_fn, minimal=False)
            return np.array(rews)

        ngridlines = int(mazehalfwidth / cellhalfwidth)
        xs = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)
        ys = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)

        if gridsize == 3:
            thetas = np.linspace(0, 2*np.pi, num=(gridsize-1)*4, endpoint=False)
            spposs = [6, 3, 2, 1, 4, 7, 8, 9]
        if gridsize == 4:
            offset = 2*np.pi/((gridsize-1)*4*2)
            thetas = np.linspace(offset, 2*np.pi + offset, num=(gridsize-1)*4, endpoint=False)
            spposs = [8, 4, 3, 2, 1, 5, 9, 13, 14, 15, 16, 12]
        for i, (theta, sppos) in enumerate(zip(thetas, spposs)):
            plt.subplot(gridsize, gridsize, sppos)
            dirv = np.array([np.cos(theta), np.sin(theta)])
            heats = get_heat_for_dir(dirv)
            heatmap = plt.pcolormesh(
                trans_arr2grid(gpts[:, 0]),
                trans_arr2grid(gpts[:, 1]),
                trans_arr2grid(heats),
                shading='nearest')
            if heat_max is not None:
                if heat_max > 0:
                    plt.clim(0, heat_max)
                elif heat_max < 0:
                    plt.clim(heat_max, -heat_max)
            plt.colorbar(heatmap)
            plt.vlines(xs, ymin=-mazehalfwidth, ymax=mazehalfwidth, colors='w', linewidths=1.5)
            plt.hlines(ys, xmin=-mazehalfwidth, xmax=mazehalfwidth, colors='w', linewidths=1.5)

        plt.tight_layout()
        return figure_to_image(fig)

    def build_gdisc_heat_fig(self, disc, cellhalfwidth=0.2, mazehalfwidth=3.0, heat_max=None, norm_fn=None, gridsize=4):
        fig = plt.figure(figsize=(int(5*3), int(4*3)))

        modparam = 0.5 / cellhalfwidth
        # obs_next
        gpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        gpts = np.array([list(pt) for pt in gpts])
        # obs
        cpts = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 101).tolist() for _ in range(2)])
        cpts = np.array([list(pt) for pt in cpts])
        cpts = np.around(cpts * modparam) / modparam
        # acts
        acts = gpts - cpts
        acts = acts/(np.max(acts)+1e-8)

        def get_heat_for_dir(dirv):
            def heat_fn(s_np, a_np, s2_np):
                with torch.no_grad():
                    assert s_np.ndim == 2 and a_np.ndim == 2 and s2_np.ndim == 2
                    s_th = torch.Tensor(norm_fn(s_np))
                    a_th = torch.Tensor(a_np)
                    s2_th = torch.Tensor(norm_fn(s2_np))
                    z_th = torch.tile(torch.Tensor(dirv).unsqueeze(0), (s_th.size(0), 1))
                    sz_th = torch.cat([s_th, z_th], 1)
                    s2z_th = torch.cat([s2_th, z_th], 1)
                    r_th = disc.discrim_net.get_reward(sz_th, a_th, s2z_th, pure_reward=True)[:, 0]
                return r_th.data.numpy()
            rews = get_heat_for_xy(cpts, acts, gpts, heat_fn, minimal=False)
            return np.array(rews)

        ngridlines = int(mazehalfwidth / cellhalfwidth)
        xs = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)
        ys = cellhalfwidth + np.linspace(-mazehalfwidth, mazehalfwidth, endpoint=False, num=ngridlines)

        if gridsize == 4:
            ptx = np.linspace(-1, 1, num=gridsize+2, endpoint=True)[1:-1]
            pty = np.linspace(-1, 1, num=gridsize+2, endpoint=True)[1:-1][::-1]
            codes = itertools.product(pty, ptx)
        for i, code in enumerate(codes):
            code = code[::-1]
            plt.subplot(gridsize, gridsize, i+1)
            heats = get_heat_for_dir(code)
            heatmap = plt.pcolormesh(
                trans_arr2grid(gpts[:, 0]),
                trans_arr2grid(gpts[:, 1]),
                trans_arr2grid(heats),
                shading='nearest')
            if heat_max is not None:
                if heat_max > 0:
                    plt.clim(0, heat_max)
                elif heat_max < 0:
                    plt.clim(heat_max, -heat_max)
            plt.colorbar(heatmap)
            plt.vlines(xs, ymin=-mazehalfwidth, ymax=mazehalfwidth, colors='w', linewidths=1.5)
            plt.hlines(ys, xmin=-mazehalfwidth, xmax=mazehalfwidth, colors='w', linewidths=1.5)

        plt.tight_layout()
        return figure_to_image(fig)

    def build_skills_space_traces_fig(self, rollouts, disc, norm_fn=None, mazehalfwidth=3.0):
        fig = plt.figure(figsize=(15, 10))

        def rep_fn(s_np, blowup=False):
            with torch.no_grad():
                assert s_np.ndim == 2
                if blowup: s_np = get_obs_from_xy(s_np)
                s_th = torch.Tensor(norm_fn(s_np))
                r_th = disc.decoder.get_code(s_th, torch.zeros_like(s_th[:, :2]))
            return r_th.data.numpy()

        def code_fn(s_np, blowup=False, return_prob=False):
            with torch.no_grad():
                assert s_np.ndim == 2
                if blowup: s_np = get_obs_from_xy(s_np)
                s_th = torch.Tensor(norm_fn(s_np))
                r_th, p_th = disc.infer_code(s_th, torch.zeros_like(s_th[:, :2]), return_prob=return_prob)
            return r_th.data.numpy(), p_th.data.numpy()

        gtspace = itertools.product(*[np.linspace(-mazehalfwidth, mazehalfwidth, 100).tolist() for _ in range(2)])
        gtspace = np.array([list(pt) for pt in gtspace])
        rpspace = rep_fn(gtspace, blowup=True)

        codes = np.array([r['code'] for r in rollouts]); colors = get_code_colors(codes)
        codes_archive = disc.get_archive_numpy(); colors_archive = get_code_colors(codes_archive)
        colors_gtspace = get_code_colors(gtspace)
        colors_gtspace[::10] = [0,0, 0, 1.0] # to be able to tell rows properly, color row boundaries black

        offset = 2*np.pi/((4-1)*4*2); thetas = np.linspace(offset, np.pi + offset, num=(4-1)*2, endpoint=False)

        plt.subplot(2, 3, 4)
        # plot inferred codes
        code_gtspace, cpbs_gts = code_fn(gtspace, blowup=True, return_prob=True)
        # multiply by blackener to account for confidence
        # cpbs_gts = (cpbs_gts - cpbs_gts.min())/(cpbs_gts.max() - cpbs_gts.min())
        # code_gtspace = code_gtspace * cpbs_gts
        cgs_bounds = [code_gtspace.min(axis=0), code_gtspace.max(axis=0)]
        # code_gtspace = -1 + 2 * (code_gtspace - cgs_bounds[0]) / (cgs_bounds[1] - cgs_bounds[0])
        # plt.scatter(gtspace[:, 0], gtspace[:, 1], 1, c=get_code_colors(code_gtspace))
        plt.imshow(trans_arr2grid(get_code_colors(code_gtspace)))

        # sampled latents
        plt.subplot(2, 3, 1)
        plt.xlim((-3.0, 3.0)); plt.ylim((-3.0, 3.0))
        for code, color in zip(codes, colors):
            if len(code) < 2: code = [code[0], 0]
            plt.scatter(code[0], code[1], s=10, color=color)
        plt.plot([cgs_bounds[0][0], cgs_bounds[1][0], cgs_bounds[1][0], cgs_bounds[0][0], cgs_bounds[0][0]],
                 [cgs_bounds[0][1], cgs_bounds[0][1], cgs_bounds[1][1], cgs_bounds[1][1], cgs_bounds[0][1]],
                 'k--', alpha=0.5)
        # archive latents
        for i, (code, color) in enumerate(zip(codes_archive, colors_archive)):
            if len(code) < 2: code = [code[0], 0]
            plt.scatter(code[0], code[1], s=25, color='black')
            plt.annotate(str(i), code[:2])

        # plot state space
        plt.subplot(2, 3, 2)
        # plt.scatter(gtspace[:, 0], gtspace[:, 1], 1, c=colors_gtspace)
        plt.imshow(trans_arr2grid(colors_gtspace))
        # plot rep space
        plt.subplot(2, 3, 3)
        plt.scatter(rpspace[:, 0], rpspace[:, 1], 1, c=colors_gtspace)

        # skills traces in state space
        plt.subplot(2, 3, 5)
        plt.xlim((-mazehalfwidth, mazehalfwidth)); plt.ylim((-mazehalfwidth, mazehalfwidth))
        for ir, rollout in enumerate(rollouts):
            obses = rollout['observations']
            plt.plot(obses[:, 0], obses[:, 1], c=colors[ir])
            plt.scatter(obses[:, 0], obses[:, 1], s=3, c='black')

        # emb traces
        plt.subplot(2, 3, 6)
        plt.scatter(rpspace[:, 0], rpspace[:, 1], 1, c='grey', alpha=0.2)

        for ir, rollout in enumerate(rollouts):
            robses = rep_fn(rollout['observations'])
            plt.plot(robses[:, 0], robses[:, 1], c=colors[ir])

        plt.tight_layout()
        return figure_to_image(fig)

def get_obs_from_xy(state_xy):
    assert state_xy.ndim == 2
    nobs = state_xy.shape[0]
    observation = np.zeros((nobs, 25))
    observation[:, 0:2] = state_xy.copy()
    observation[:, 3:5] = state_xy.copy()
    return observation

def get_heat_for_xy(obs_xy, act, nobs_xy, heat_fn, minimal=False):
    obs = get_obs_from_xy(obs_xy) if not minimal else obs_xy
    obs_next = get_obs_from_xy(nobs_xy) if not minimal else nobs_xy
    act = act
    return heat_fn(obs, act, obs_next)

def trans_arr2grid(arr):
    arr_shape = list(arr.shape)
    nelem, rshape = arr_shape[0], arr_shape[1:]
    ssqre = int(np.sqrt(nelem))
    assert ssqre == np.sqrt(nelem)
    arr = np.reshape(arr, [ssqre, ssqre] + rshape)
    arr = arr[:, ::-1, ...]
    perm_axes = np.arange(arr.ndim); perm_axes[0: 2] = [1, 0]
    arr = np.transpose(arr, perm_axes)
    return arr

def get_code_colors(points, min_point=(-1.5, -1.5), max_point=(1.5, 1.5)):
    points = np.array(points)
    if points.shape[1] > 2:
        # print('Considering first 2 dims for skill scatter plot')
        points = points[:, :2]
    min_point = np.array(min_point)
    max_point = np.array(max_point)
    colors = (points - min_point) / (max_point - min_point)
    colors = np.hstack((
        colors,
        (2 - np.sum(colors, axis=1, keepdims=True)) / 2,
    ))
    colors = np.clip(colors, 0, 1)
    colors = np.c_[colors, np.full(len(colors), 1.0)]
    return colors

def calc_mvn_params(vec_g):
    m_g = np.mean(vec_g, axis=0)
    S_g = np.cov(vec_g, rowvar=False) + np.eye(m_g.shape[0])*1e-8
    return m_g, S_g

# https://gist.github.com/ChuaCheowHuan/18977a3e77c0655d945e8af60633e4df
from scipy.linalg import cho_factor, cho_solve
def kl_mvn(to, fr):
    m_to, S_to = to
    m_fr, S_fr = fr
    d = m_fr - m_to
    c, lower = cho_factor(S_fr)
    def solve(B):
        return cho_solve((c, lower), B)
    def logdet(S):
        return np.linalg.slogdet(S)[1]
    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.

def logkl_mvn(*args, **kwargs):
    return np.log(kl_mvn(*args, **kwargs))

def js_mvn(to, fr):
    return (kl_mvn(to, fr) + kl_mvn(fr, to))/2
