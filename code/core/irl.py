from my_utils import *
from core_nn.nn_irl import * 
# from core_nn.nn_old import *
import h5py 

""" MaxEnt-IRL. I.e., Adversarial IL with linear loss function. """
class IRL(): 
    def __init__(self, state_dim, action_dim, args, initialize_net=True, load_verbose=True):
        self.mini_batch_size = args.mini_batch_size 
        self.gp_lambda = args.gp_lambda  
        self.gp_alpha = args.gp_alpha  
        self.gp_center = args.gp_center 
        self.gp_lp = args.gp_lp 
        self.state_dim = state_dim 
        self.action_dim = action_dim 
        self.gamma = args.gamma

        self.load_demo_list(args, verbose=load_verbose)
        if initialize_net:
            self.initilize_nets(args) 
            
    def initilize_nets(self, args):   
        self.discrim_net = Discriminator(self.state_dim, self.action_dim, hidden_size=args.hidden_size, activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)  

    def load_demo_list(self, args, verbose=True):

        self.traj_name = "%s_v%d_type%d" % (args.env_name, args.v_data, args.c_data)
        basedir = '/'.join(__file__.split('/')[:-3])
        traj_filename = "%s/imitation_data/STRAT_h5/%s_v%d.h5" % (basedir, args.env_name, args.v_data)
        hf = h5py.File(traj_filename, 'r')
        expert_states_all = hf.get('expert_states')[:]
        expert_actions_all = hf.get('expert_actions')[:]
        expert_masks_all = hf.get('expert_masks')[:]
        expert_rewards_all = hf.get('expert_rewards')[:]
        expert_codes_all = hf.get('expert_codes')[:]
        expert_ids_all = hf.get('expert_ids')[:]

        step_num = expert_masks_all.shape[0]
        traj_num = step_num - np.sum(expert_masks_all)   
        m_return = np.sum(expert_rewards_all) / traj_num
        self.m_return_list = [ m_return ]
        if verbose:
            print("TRAJ is loaded from %s with traj_num %s, data_size %s steps, and average return %s" % \
                (colored(traj_filename, p_color), colored(traj_num, p_color), colored(expert_masks_all.shape[0] , p_color), \
                colored( "%.2f" % (m_return), p_color )))

        # split into test and non test based on c_data
        terminal_idxs = np.where(expert_masks_all == 0)[0] + 1
        eidxs_all = np.transpose([[0] + terminal_idxs[:-1].tolist(), terminal_idxs])

        # Make disjoint train, test
        # split per dim of code into blocks
        n_segs = args.c_segs
        cndim = expert_codes_all.shape[1]
        segs_list = [[] for _ in range(n_segs**cndim)]

        sids = np.zeros_like(expert_codes_all[:, 0], dtype=int)
        for cidim in range(cndim):
            cx = expert_codes_all[:, cidim]
            cx = (cx - cx.min()) / (cx.max() - cx.min() + 1e-8)
            cx = np.floor(cx * n_segs)
            sids = sids*n_segs + (cx).astype(int)

        for i, sid in enumerate(sids):
            segs_list[sid].append(i)

        if args.c_data == 0:
            segs_train = segs_list
            segs_teval = segs_list
        elif args.c_data == 1:
            segs_train = segs_list[0::2]
            segs_teval = segs_list[1::2]
        elif args.c_data == 2:
            segs_train = segs_list[1::2]
            segs_teval = segs_list[0::2]
        else:
            raise NotImplementedError
        tids_train = np.concatenate(segs_train, axis=0)
        tids_teval = np.concatenate(segs_teval, axis=0)

        # split teval into test and val randomly
        np.random.RandomState(42).shuffle(tids_teval)
        split_len = int(0.75 * tids_teval.shape[0])
        tids_test, tids_val = np.sort(tids_teval[:split_len]), np.sort(tids_teval[split_len:])

        if verbose:
            print(f"No of trajs: Train {colored(len(tids_train), p_color)} Val {colored(len(tids_val), p_color)} Test {colored(len(tids_test), p_color)}")
            # print(tids_train)
            # print(tids_val)
            # print(tids_test)

        eidxs_train = eidxs_all[tids_train]
        eidxs_val = eidxs_all[tids_val]
        eidxs_test = eidxs_all[tids_test]

        # stored for use in eval
        self.expert_states_train = [expert_states_all[st:en, :] for st, en in eidxs_train]
        self.expert_actions_train = [expert_actions_all[st:en, :] for st, en in eidxs_train]
        self.expert_masks_train = [expert_masks_all[st:en] for st, en in eidxs_train]
        self.expert_ids_train = [expert_ids_all[st:en] for st, en in eidxs_train]
        # self.expert_rewards_train = [expert_rewards_all[st:en] for st, en in eidxs_train]

        self.expert_states_val = [expert_states_all[st:en, :] for st, en in eidxs_val]
        self.expert_actions_val = [expert_actions_all[st:en, :] for st, en in eidxs_val]
        self.expert_masks_val = [expert_masks_all[st:en] for st, en in eidxs_val]
        self.expert_ids_val = [expert_ids_all[st:en] for st, en in eidxs_val]
        # self.expert_rewards_val = [expert_rewards_all[st:en] for st, en in eidxs_val]

        self.expert_states_test = [expert_states_all[st:en, :] for st, en in eidxs_test]
        self.expert_actions_test = [expert_actions_all[st:en, :] for st, en in eidxs_test]
        self.expert_masks_test = [expert_masks_all[st:en] for st, en in eidxs_test]
        self.expert_ids_test = [expert_ids_all[st:en] for st, en in eidxs_test]
        # self.expert_rewards_test = [expert_rewards_all[st:en] for st, en in eidxs_test]

        real_state_tensor_raw = torch.FloatTensor(np.concatenate(self.expert_states_train, axis=0)).to(device_cpu) 
        real_action_tensor_raw = torch.FloatTensor(np.concatenate(self.expert_actions_train, axis=0)).to(device_cpu) 
        real_mask_tensor_raw = torch.FloatTensor(np.concatenate(self.expert_masks_train, axis=0)).to(device_cpu) 
        real_worker_tensor_raw = torch.LongTensor(np.concatenate(self.expert_ids_train, axis=0)).to(device_cpu) 
        data_size_raw = real_state_tensor_raw.size(0) 
        self.worker_num = torch.unique(real_worker_tensor_raw).size(0) # much cleaner code

        # check for explanation under update_discriminator in gsdr.py
        assert real_mask_tensor_raw[data_size_raw - 1] == 0
        valid_nstate_idxs = torch.where(real_mask_tensor_raw == 1)[0]
        self.real_state_tensor = real_state_tensor_raw[valid_nstate_idxs, :]
        self.real_action_tensor = real_action_tensor_raw[valid_nstate_idxs, :]
        self.real_nstate_tensor = real_state_tensor_raw[valid_nstate_idxs + 1, :]
        self.real_worker_tensor = real_worker_tensor_raw[valid_nstate_idxs]
        self.data_size = self.real_state_tensor.size(0)
        # add ids of the form 0-N to demos
        _wid2zid = {wid: zid for zid, wid in enumerate(torch.unique(self.real_worker_tensor).data.numpy())}
        _rwtf = torch.Tensor([_wid2zid[x] for x in self.real_worker_tensor.data.numpy()])
        self.real_worker_tensor = torch.Tensor.long(_rwtf)
        self.tdmap = torch.zeros(self.worker_num, self.data_size)
        self.tdmap[self.real_worker_tensor, torch.arange(self.data_size)] = 1
        self.ntdmap = self.tdmap / torch.sum(self.tdmap, dim=1, keepdim=True)

        if verbose:
            print("Total data pairs: %s, K %s, state dim %s, action dim %s, a min %s, a_max %s" % \
                (colored(self.real_state_tensor.size(0), p_color), colored(self.worker_num, p_color), \
                colored(self.real_state_tensor.size(1), p_color), colored(self.real_action_tensor.size(1), p_color), 
                colored(torch.min(self.real_action_tensor).numpy(), p_color), colored(torch.max(self.real_action_tensor).numpy(), p_color)   \
                ))

    def compute_reward(self, states, actions, next_states=None, masks=None):
        return self.discrim_net.get_reward(states, actions)

    def update_curriculum(self, *args, **kwargs):
        pass

    def save_model(self, path):
        torch.save({
            'discrim_net': self.discrim_net.state_dict(),
        }, path)

    def load_model(self, path):
        # self.policy_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        pass

    def print_obs_stats(self):
        exp_states = self.real_state_tensor.data.numpy()
        def get_arr_str(arr):
            sstr = 'np.array(['
            for elem in arr:
                sstr += str(elem) + ', '
            sstr += '])'
            return sstr
        print(get_arr_str(np.mean(exp_states, axis=0)))
        print(get_arr_str(np.var(exp_states, axis=0)))

    def normalize_expert_data(self, mean, std):
        mean_th = torch.FloatTensor(mean).unsqueeze(0)
        std_th = torch.FloatTensor(std).unsqueeze(0)
        self.real_state_tensor = torch.clip((self.real_state_tensor - mean_th) / std_th, -10, 10)

    def index_sampler(self, offset=0, batch_size=None):
        if batch_size is None:
            batch_size = self.mini_batch_size
        return torch.randperm(self.data_size-offset)[0:batch_size].to(device_cpu)

    def get_bc_data(self, index):
        s_expert = self.real_state_tensor[index, :].to(device)
        a_expert = self.real_action_tensor[index, :].to(device)
        return s_expert, a_expert

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)    
        x_fake = self.discrim_net.get_reward(s_fake, a_fake) 
                
        loss_real = x_real.mean()
        loss_fake = x_fake.mean() 
        loss = -(loss_real - loss_fake)
    
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step() 
            
    def gp_regularizer(self, sa_real, sa_fake):
        if self.gp_lambda == 0:
            return 0

        real_data = sa_real.data
        fake_data = sa_fake.data
                                       
        if real_data.size(0) < fake_data.size(0):
            idx = np.random.permutation(fake_data.size(0))[0: real_data.size(0)]
            fake_data = fake_data[idx, :]
        else: 
            idx = np.random.permutation(real_data.size(0))[0: fake_data.size(0)]
            real_data = real_data[idx, :]
            
        if self.gp_alpha == "mix":
            alpha = torch.rand(real_data.size(0), 1).expand(real_data.size()).to(device)
            x_hat = alpha * real_data + (1 - alpha) * fake_data
        elif self.gp_alpha == "real":
            x_hat = real_data
        elif self.gp_alpha == "fake":
            x_hat = fake_data 

        x_hat_out = self.discrim_net(x_hat.to(device).requires_grad_())
        gradients = torch.autograd.grad(outputs=x_hat_out, inputs=x_hat, \
                        grad_outputs=torch.ones(x_hat_out.size()).to(device), \
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        if self.gp_lp:
            return ( torch.max(0, gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda
        else:
            return ((gradients.norm(2, dim=1) - self.gp_center) ** 2).mean() * self.gp_lambda

    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None :
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(self.real_state_tensor.to(device), self.real_action_tensor.to(device), self.real_worker_tensor.unsqueeze(-1).to(device))

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch, w_batch) in enumerate(train_loader):
                count = count + 1       

                action_mean, _, _ = policy_net( s_batch )
                loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()    ##

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
        

""" GAIL """
class GAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args)
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                   
    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions)) # minimize agent label score. 
        
    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   
                
        adversarial_loss = torch.nn.BCEWithLogitsLoss() 
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake
        
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" VAIL """
class VAIL(IRL):
    def __init__(self, state_dim, action_dim, args):
        super().__init__(state_dim, action_dim, args) 
        self.vdb_ic = 0.5   
        self.vdb_beta = 0    
        self.vdb_alpha_beta = 1e-5  
        self.bce_negative = args.bce_negative   # Code should be cleaner if we just extend GAIL class.
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                       
    def compute_reward(self, states, actions, next_states=None, masks=None):
        if self.bce_negative:
            return F.logsigmoid(-self.discrim_net.get_reward(states, actions))   # maximize expert label score. 
        else:
            return -F.logsigmoid(-self.discrim_net.get_reward(states, actions)) # minimize agent label score. 
        
    def initilize_nets(self, args):   
        self.discrim_net = VDB_discriminator(self.state_dim, self.action_dim, encode_dim=128, hidden_size=args.hidden_size, activation=args.activation, clip=args.clip_discriminator).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)  

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real, z_real_mean, z_real_logstd = self.discrim_net.get_full(s_real, a_real)  
        x_fake, z_fake_mean, z_fake_logstd = self.discrim_net.get_full(s_fake, a_fake)

        adversarial_loss = torch.nn.BCEWithLogitsLoss() 
        label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(x_real, label_real)
        loss_fake = adversarial_loss(x_fake, label_fake)
        loss = loss_real + loss_fake
    
        ## compute KL from E(z|x) = N(z_mean, z_std) to N(0,I). #sum across dim z, then average across batch size.
        kl_real = 0.5 * ( -z_real_logstd + torch.exp(z_real_logstd) + z_real_mean**2 - 1).sum(dim=1).mean()  
        kl_fake = 0.5 * ( -z_fake_logstd + torch.exp(z_fake_logstd) + z_fake_mean**2 - 1).sum(dim=1).mean()  
        bottleneck_reg = 0.5 * (kl_real + kl_fake) - self.vdb_ic

        loss += self.vdb_beta * bottleneck_reg
        self.vdb_beta = max(0, self.vdb_beta + self.vdb_alpha_beta * bottleneck_reg.detach().cpu().numpy())

        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" AIRL """
class AIRL(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater=None):
        super().__init__(state_dim, action_dim, args)
        self.policy_updater = policy_updater
        self.label_real = 1 
        self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
              
    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   
            
        ent_coef = self.policy_updater.entropy_coef.detach()   
        log_probs_real = self.policy_updater.policy_net.get_log_prob(s_real, a_real).detach()
        log_probs_fake = self.policy_updater.policy_net.get_log_prob(s_fake, a_fake).detach()

        adversarial_loss = torch.nn.CrossEntropyLoss() 
        label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
        loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
        loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
        loss = loss_real + loss_fake
        
        """ gradient penalty regularization """
        loss += self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))

        """ Update """
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
""" InfoGAIL """
class InfoGAIL(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater=None):
        super().__init__(state_dim, action_dim, args)
        self.encode_dim = args.encode_dim
        self.encode_cont = args.encode_cont
        self.encode_sampling = args.encode_sampling
        self.info_coef = args.info_coef 
        self.loss_type = args.info_loss_type.lower() 
        self.policy_updater = policy_updater

        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0 
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1 
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid
                       
        self.initilize_posterior_nets(args) 

    def initilize_posterior_nets(self, args):
        self.posterior_net = Posterior(input_dim=self.state_dim + self.action_dim, encode_dim=self.encode_dim, encode_cont=self.encode_cont, hidden_size=args.hidden_size, activation=args.activation).to(device)
        self.optimizer_posterior = torch.optim.Adam(self.posterior_net.parameters(), lr=args.learning_rate_d)  

    def save_model(self, path):
        torch.save({
            'discrim_net': self.discrim_net.state_dict(),
            'posterior_net': self.posterior_net.state_dict(),
        }, path)

    def compute_reward(self, states, actions, next_states=None, masks=None, codes=None):
        disclogit = self.discrim_net.get_reward(states, actions)
        if self.loss_type == "bce":    # binary_cross entropy. corresponding to standard InfoGAIL
            if self.bce_negative:
                rwd =  F.logsigmoid(-disclogit)   # maximize expert label score. 
            else:
                rwd =  -F.logsigmoid(-disclogit)
        else:   # Wasserstein variant of InfoGAIL. 
            rwd =  disclogit
        return rwd

    def compute_posterior_reward(self, states, actions, latent_codes, next_states=None, masks=None):
        reward_p = self.posterior_net.get_logposterior(states, actions, latent_codes) 
        return self.info_coef * reward_p

    def sample_code(self, train=False):
        if not self.encode_cont:
            return torch.randint(0, self.encode_dim, size=(1,1))  
        else:
            if self.encode_sampling == 'uniform':
                return torch.rand(size=(1,self.encode_dim))*2 - 1
            elif self.encode_sampling == 'normal':
                return torch.randn(size=(1,self.encode_dim))
            elif self.encode_sampling == 'spherical':
                sampled_code = torch.randn(size=(1,self.encode_dim))
                return normalize_tens(sampled_code)
            elif self.encode_sampling == 'cyclic':
                if not hasattr(self, 'anchors'):
                    assert self.encode_dim == 2
                    ls = np.linspace(-0.8, 0.8, 5)
                    self.anchors = torch.FloatTensor(np.array([[i, j] for i in ls for j in ls]))
                    self.anchors_len = len(self.anchors)
                    self.anchor_id = 0
                noise_code = (torch.rand(size=(1, self.encode_dim))*2-1)*0.2
                sampled_code = self.anchors[self.anchor_id].view(1, self.encode_dim) + noise_code
                self.anchor_id = (self.anchor_id + 1) % self.anchors_len
                return sampled_code

    @property
    def sampled_archive_id(self):
        return -1

    def get_bc_data(self, index):
        s_expert = self.real_state_tensor[index, :].to(device)
        a_expert = self.real_action_tensor[index, :].to(device)
        post_net_inp = torch.cat([s_expert, a_expert], 1)
        z_expert = self.posterior_net(post_net_inp)[:, :self.encode_dim].detach()
        sz_expert = torch.cat([s_expert, z_expert], 1)
        return sz_expert, a_expert

    def update_discriminator(self, batch, index, total_step=0):
        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)

        x_real = self.discrim_net.get_reward(s_real, a_real)         
        x_fake = self.discrim_net.get_reward(s_fake, a_fake)   

        if self.loss_type == "linear":
            loss_real = x_real.mean()
            loss_fake = x_fake.mean() 
            loss = -(loss_real - loss_fake)

        elif self.loss_type == "bce":    # Binary_cross entropy for GAIL-like variant.
            adversarial_loss = torch.nn.BCEWithLogitsLoss() 
            label_real = Variable(FloatTensor(x_real.size(0), 1).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(FloatTensor(x_fake.size(0), 1).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss(x_real, label_real)
            loss_fake = adversarial_loss(x_fake, label_fake)
            loss = loss_real + loss_fake
            
        elif self.loss_type == "ace":    # AIRL cross entropy for AIRL-like variant.
            ent_coef = self.policy_updater.entropy_coef.detach()   
            log_probs_real = self.policy_updater.policy_net.get_log_prob(s_real, a_real).detach()
            log_probs_fake = self.policy_updater.policy_net.get_log_prob(s_fake, a_fake).detach()

            adversarial_loss = torch.nn.CrossEntropyLoss() 
            label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
            label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
            loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
            loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
            loss = loss_real + loss_fake

        gp_loss = self.gp_regularizer(torch.cat((s_real, a_real), 1), torch.cat((s_fake, a_fake), 1))
        loss += gp_loss
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()
        
        """ update posterior of infogail """
        if not self.encode_cont:
            latent_codes_fake = torch.LongTensor(np.stack(batch.latent_code)).to(device)    # Label: scalar value in range [0, code_dim-1] 
            latent_score = self.posterior_net( torch.cat((s_fake, a_fake), 1))
            posterior_loss = torch.nn.CrossEntropyLoss()
            p_loss = posterior_loss(latent_score, latent_codes_fake.squeeze())
        else:
            # instead of MSE loss we directly use logposterior here
            latent_codes_fake = torch.FloatTensor(np.stack(batch.latent_code)).view(-1, self.encode_dim).to(device)
            latent_codes_lprob = self.posterior_net.get_logposterior(s_fake, a_fake, latent_codes_fake)
            p_loss = -torch.mean(latent_codes_lprob)

        self.optimizer_posterior.zero_grad()
        p_loss.backward()
        self.optimizer_posterior.step()

        return {
            'disc_loss': loss.item(),
            'disc_reward/min': x_fake.min().item(),
            'disc_reward/max': x_fake.max().item(),
            'code_loss': p_loss.item(),
        }

    ## Re-define BC function because it needs context variable as input. 
    def behavior_cloning(self, policy_net=None, learning_rate=3e-4, bc_step=0):
        if bc_step <= 0 or policy_net is None :
            return

        bc_step_per_epoch = self.data_size / self.mini_batch_size 
        bc_epochs = math.ceil(bc_step/bc_step_per_epoch)

        train = data_utils.TensorDataset(
            self.real_state_tensor.to(device),
            self.real_action_tensor.to(device),
        )

        train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size)
        optimizer_pi_bc = torch.optim.Adam(policy_net.parameters(), lr=learning_rate) 
        
        count = 0
        print("Behavior cloning: %d epochs... " % bc_epochs)
        t0 = time.time()
        for epoch in range(0, bc_epochs):
            for batch_idx, (s_batch, a_batch) in enumerate(train_loader):
                count = count + 1       

                w_batch = torch.cat([self.sample_code() for _ in range(s_batch.size(0))], 0)
                if not self.encode_cont:
                    # We use sampled context as context variable here. 
                    latent_codes_onehot = torch.FloatTensor(s_batch.size(0), self.encode_dim).to(device)
                    latent_codes_onehot.zero_()
                    latent_codes_onehot.scatter_(1, w_batch, 1)  #should have size [batch_size, num_worker]
                else:
                    latent_codes_onehot = w_batch
                s_batch = torch.cat((s_batch, latent_codes_onehot), 1)  # input of the policy function. 

                # action_mean, _, _ = policy_net( s_batch )
                # loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()
                log_probs = policy_net.get_log_prob(s_batch, a_batch)
                loss = -log_probs.mean()

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
        