
from my_utils import *

from core.irl import IRL
from core_nn.nn_gsdr import SNDecoder
from core_nn.nn_gsdr import PairDiscriminator


class GSDR(IRL):
    def __init__(self, state_dim, action_dim, args, policy_updater=None):
        super().__init__(state_dim, action_dim, args, initialize_net=False)
        self.encode_dim = args.encode_dim
        self.encode_cont = args.encode_cont
        self.encode_sampling = args.encode_sampling
        self.policy_updater = policy_updater

        self.info_loss_type = args.info_loss_type
        self.bce_negative = args.bce_negative
        if self.bce_negative:
            self.label_real = 0
            self.label_fake = 1   # with BCEloss: E_{pi}[log D] + E_{ex}[log(1-D)] with D = sigmoid
        else:
            self.label_real = 1
            self.label_fake = 0   # with BCEloss: E_{ex}[log D] + E_{pi}[log(1-D)] with D = sigmoid

        assert self.encode_cont

        self.initilize_posterior_nets(args)

    def initilize_posterior_nets(self, args):
        self.regularize_decoder = args.reg_dec
        self.specnorm_decoder = args.sn_dec

        self.ac_dec = args.ac_dec
        self.sym_rew = args.sym_rew
        self.cond_rew = args.cond_rew

        self.dl_type = args.dl_type
        self.dl_ztype = args.dl_ztype
        self.dl_scale = args.dl_scale
        self.dl_l2m = args.dl_l2m

        self.cns_cf = args.cns_cf
        self.rep_cf = args.rep_cf
        self.loc_cf = args.loc_cf

        self.lr_dk = args.lr_dk
        self.dr_cc = args.dr_cc

        # discriminator
        self.discrim_net = PairDiscriminator(
            self.state_dim + (self.encode_dim if self.cond_rew else 0),
            self.action_dim,
            use_action_not_next_state=args.ac_rew,
            hidden_size=args.hidden_size,
            activation=args.activation,
            clip=args.clip_discriminator,
            shaped_reward=args.shaped_reward,
            offset_reward=args.offset_reward,
            symmetric=args.sym_rew,
            gamma=args.gamma,
            ).to(device)
        self.optimizer_discrim = torch.optim.Adam(self.discrim_net.parameters(), lr=args.learning_rate_d)
        self.lri_d = args.learning_rate_d

        # posterior
        self.decoder = SNDecoder(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            encode_dim=self.encode_dim,
            use_action=self.ac_dec,
            hidden_size=args.hs_emb,
            activation=args.actv_emb,
            specnorm=self.specnorm_decoder,
            specnorm_k=self.dl_scale,
            squash='uniform' in args.encode_sampling).to(device)
        opt_params = [
            {'params': self.decoder.parameters(), 'lr': args.lr_p, 'weight_decay': args.wd_p},
        ]
        # optimizer
        self.optimizer_posterior = torch.optim.Adam(opt_params)
        self.p_step = args.p_step
        self.lri_p = args.lr_p

        # constraints
        self.loglambda_dual = nn.Parameter(torch.ones(())*np.log(args.dl_linit))
        self.slack_dual = torch.ones(())*args.dl_slack
        self.optimizer_dual = torch.optim.Adam([self.loglambda_dual], lr=args.dl_llr)
        self.lri_dll = args.dl_llr

        self.disc_reward_frac = 0
        self.post_reward_frac = 1
        self.progress = 0.0
        self.sampled_archive_id = -1

        self.reinit_prev = -1
        self.reinitialize_misc()

    def get_archive_numpy(self):
        return self.zdemos_detached.data.numpy()

    def reinitialize_misc(self):
        self.zdemos_detached = self.infer_codes_for_demos().detach()

    def infer_codes_for_demos(self, sample_w_reparam=False):
        ### simple mean across tuples
        # zreal_each = self.infer_code(self.real_state_tensor)
        # zreal_dcode = torch.mm(self.ntdmap, zreal_each)

        ### weighted mean with std
        zreal_mean, zreal_logstd = self.decoder._get_gaussian_params(self.real_state_tensor, self.real_action_tensor)
        zreal_invar = torch.exp(zreal_logstd) ** -2
        zreal_dvar = torch.mm(self.tdmap, zreal_invar) ** -1
        zreal_dmean = torch.mm(self.tdmap, zreal_mean * zreal_invar) * zreal_dvar
        if sample_w_reparam:
            # use sample as code
            epsilon = torch.FloatTensor(zreal_dmean.size()).data.normal_(0, 1).to(device)
            zreal_dcode = zreal_dmean + zreal_dvar.pow(0.5)*epsilon
        else:
            # use mean as code
            zreal_dcode = zreal_dmean

        if self.decoder.squash:
            zreal_dcode = torch.tanh(zreal_dcode)
        return zreal_dcode

    def save_model(self, path):
        torch.save({
            'discrim_net': self.discrim_net.state_dict(),
            'decoder': self.decoder.state_dict(),
            'loglambda_dual': self.loglambda_dual,
            'zdemos': getattr(self, 'zdemos_detached', None),
        }, path)

    def load_model(self, path):
        # self.policy_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        pass

    def normalize_expert_data(self, mean, std):
        mean_th = torch.FloatTensor(mean).unsqueeze(0)
        std_th = torch.FloatTensor(std).unsqueeze(0)
        self.real_state_tensor = torch.clip((self.real_state_tensor - mean_th) / std_th, -10, 10)
        self.real_nstate_tensor = torch.clip((self.real_nstate_tensor - mean_th) / std_th, -10, 10)

    def update_curriculum(self, progress):
        self.progress = np.clip(progress, 0, 1)

        if self.lr_dk == 1:
            # schedule for all lrs
            update_linear_schedule(self.optimizer_discrim, self.progress, self.lri_d)
            update_linear_schedule(self.optimizer_posterior, self.progress, self.lri_p)
            update_linear_schedule(self.optimizer_dual, self.progress, self.lri_dll)

        # schedule for rewards from discriminator
        if self.dr_cc == 0:
            self.disc_reward_frac = 0
        elif self.dr_cc == 1:
            self.disc_reward_frac = 1
        elif self.dr_cc > 0:
            # OLD: disc reward decreases and post reward increases
            # self.disc_reward_frac = (1 - self.progress)**self.dr_cc
            # disc reward is set to constant
            self.disc_reward_frac = self.dr_cc
        else:
            # -dr_cc treated as half life for disc reward
            self.disc_reward_frac = 0.5 ** (self.progress/-self.dr_cc)
        self.post_reward_frac = 1 - self.disc_reward_frac

    def compute_reward(self, states, actions, next_states=None, masks=None, codes=None):
        if self.cond_rew:
            # disc_inp = torch.cat([states, codes, actions, next_states, codes], 1)
            disc_inp = torch.cat([states, codes, actions], 1)
        else:
            # disc_inp = torch.cat([states, actions, next_states], 1)
            disc_inp = torch.cat([states, actions], 1)
        # here r might be clamped by tanh so use full reward not pure reward
        disclogit = self.discrim_net(disc_inp)

        if self.info_loss_type == 'ace':
            # logit in AIRL = r - ent logpi
            ent_coef = self.policy_updater.entropy_coef.detach()
            log_probs = self.policy_updater.policy_net.get_log_prob(torch.cat([states, codes], 1), actions).detach()
            disclogit = disclogit - ent_coef*log_probs

        # scaling to match InfoGAIL
        rwd =  -F.logsigmoid(-disclogit)
        return rwd

    def compute_posterior_reward(self, states, actions, latent_codes, next_states=None, masks=None):
        post_ll = self.decoder.get_logprob(states, actions, latent_codes)
        return post_ll * self.post_reward_frac

    def infer_code(self, states, actions, return_prob=False):
        code = self.decoder.get_code(states, actions)
        if return_prob:
            prob = torch.exp(self.decoder.get_logprob(states, actions, code))
            return code, prob
        return code

    def sample_code(self, train=False):
        if self.encode_sampling == 'uniform':
            sampled_code = torch.rand(size=(1, self.encode_dim))*2 - 1
        elif self.encode_sampling == 'normal':
            sampled_code = torch.randn(size=(1, self.encode_dim))
        elif self.encode_sampling == 'cyclic':
            if not hasattr(self, 'anchors'):
                assert self.encode_dim == 2
                self.anchors_len = 25
                self.anchors = torch.FloatTensor(np.linspace(0, 2*np.pi, self.anchors_len, endpoint=False))
                self.anchor_id = 0
            noise_code_angle = (torch.rand(size=(1,))*2-1)*((2*np.pi)/(self.anchors_len*2))
            sampled_code_angle = self.anchors[self.anchor_id].view(1) + noise_code_angle
            self.anchor_id = (self.anchor_id + 1) % self.anchors_len
            sampled_code = torch.stack([torch.cos(sampled_code_angle), torch.sin(sampled_code_angle)], dim=1)
        elif '+demos' in self.encode_sampling:
            # if train or True:
            if train and self.sampled_archive_id == -1:
                zidxs = torch.randint(0, self.worker_num, size=(1,))
                sampled_code = self.zdemos_detached[zidxs]
                self.sampled_archive_id = zidxs[0].item()
            else:
                # durin eval sampling is Uniform or Normal
                if 'uniform' in self.encode_sampling:
                    sampled_code = torch.rand(size=(1, self.encode_dim))*2 - 1
                elif 'normal' in self.encode_sampling:
                    sampled_code = torch.randn(size=(1, self.encode_dim))
                else:
                    raise NotImplementedError
                self.sampled_archive_id = -1
        else:
            raise NotImplementedError
        return sampled_code

    def get_bc_data(self, index):
        s_expert = self.real_state_tensor[index, :].to(device)
        a_expert = self.real_action_tensor[index, :].to(device)
        z_expert_dmean = self.infer_codes_for_demos().detach()
        z_expert = torch.mm(self.tdmap.T[index, :], z_expert_dmean)
        sz_expert = torch.cat([s_expert, z_expert], 1)
        return sz_expert, a_expert

    def update_discriminator(self, batch, index, total_step=0):
        # Here, mask is if the episode terminated in that step indicating that
        # the next state is actually an init state. This is for the stored demos.
        # Also, we calculate the next state by accessing index + 1. Other than overflow,
        # at index == self.data_size-1, then there is no next state.
        # To account for this we set/assert the mask value to zero to ignore it in the computation.
        # This is done in class init function.
        # m_real = self.real_mask_tensor[index].to(device)
        # s_real = self.real_state_tensor[index, :].to(device)
        # a_real = self.real_action_tensor[index, :].to(device)
        # New: Alternatively, one can remove all the tuples where nstate is invalid
        # This is worth doing as the next state must always be inferred for code in AIRL-like reward

        s_real = self.real_state_tensor[index, :].to(device)
        a_real = self.real_action_tensor[index, :].to(device)
        ns_real = self.real_nstate_tensor[index, :].to(device)

        if '+demos' in self.encode_sampling:
            z_real = self.zdemos_detached[self.real_worker_tensor[index], :].to(device)
        else:
            z_real_dmean = self.infer_codes_for_demos().detach()
            z_real = torch.mm(self.tdmap.T[index, :], z_real_dmean)

        # debugging
        # z_real_each = self.infer_code(self.real_state_tensor)
        # dranges = []
        # for didxs in self.tdmap:
        #     drange = z_real_each[didxs == 1, :].amax(dim=0) - z_real_each[didxs == 1, :].amin(dim=0)
        #     dranges.append(drange.data.numpy())
        # print(np.min(dranges, axis=0), np.max(dranges, axis=0))

        s_fake = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_fake = torch.FloatTensor(np.stack(batch.action)).to(device)
        ns_fake = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        z_fake = torch.FloatTensor(np.stack(batch.latent_code)).view(-1, self.encode_dim).to(device)

        # filter z to include those where archive id != -1
        if '+demos' in self.encode_sampling:
            idxs_w_dids = torch.FloatTensor(np.stack(batch.latent_archive_id)) != -1
            s_fake = s_fake[idxs_w_dids]
            a_fake = a_fake[idxs_w_dids]
            ns_fake = ns_fake[idxs_w_dids]
            z_fake = z_fake[idxs_w_dids]

        if self.cond_rew:
            # shuffle z to ensure reward fn is conditioned well on z and does not collapse
            s_fake = torch.cat([s_fake, s_fake])
            a_fake = torch.cat([a_fake, a_fake])
            z_fake = torch.cat([z_fake, z_fake[torch.randperm(z_fake.size()[0]), :]])
            ns_fake = torch.cat([ns_fake, ns_fake])

        # z concatenated to the end of state vector, not the start
        sz_real = torch.cat([s_real, z_real], dim=1)
        sz_fake = torch.cat([s_fake, z_fake], dim=1)
        nsz_real = torch.cat([ns_real, z_real], dim=1)
        nsz_fake = torch.cat([ns_fake, z_fake], dim=1)

        if self.cond_rew:
            # disc_inp_real = torch.cat([sz_real, a_real, nsz_real], 1)
            # disc_inp_fake = torch.cat([sz_fake, a_fake, nsz_fake], 1)
            disc_inp_real = torch.cat([sz_real, a_real], 1)
            disc_inp_fake = torch.cat([sz_fake, a_fake], 1)
        else:
            # disc_inp_real = torch.cat([s_real, a_real, ns_real], 1)
            # disc_inp_fake = torch.cat([s_fake, a_fake, ns_fake], 1)
            disc_inp_real = torch.cat([s_real, a_real], 1)
            disc_inp_fake = torch.cat([s_fake, a_fake], 1)

        # reward is shaped
        x_real = self.discrim_net(disc_inp_real)
        x_fake = self.discrim_net(disc_inp_fake)

        log_probs_real = self.policy_updater.policy_net.get_log_prob(sz_real, a_real).detach()
        log_probs_fake = self.policy_updater.policy_net.get_log_prob(sz_fake, a_fake).detach()
        label_real = Variable(LongTensor(x_real.size(0)).fill_(self.label_real), requires_grad=False).to(device)
        label_fake = Variable(LongTensor(x_fake.size(0)).fill_(self.label_fake), requires_grad=False).to(device)
        ent_coef = self.policy_updater.entropy_coef.detach()

        if self.info_loss_type == "bce":
            adversarial_loss = torch.nn.BCEWithLogitsLoss()
            loss_real = adversarial_loss(x_real[:, 0], label_real.float())
            loss_fake = adversarial_loss(x_fake[:, 0], label_fake.float())
            logits_real = x_real.detach()
            logits_fake = x_fake.detach()
        elif self.info_loss_type == "ace":
            adversarial_loss = torch.nn.CrossEntropyLoss() 
            loss_real = adversarial_loss(torch.cat((ent_coef * log_probs_real, x_real), 1), label_real)
            loss_fake = adversarial_loss(torch.cat((ent_coef * log_probs_fake, x_fake), 1), label_fake)
            logits_real = (x_real - ent_coef*log_probs_real).detach()
            logits_fake = (x_fake - ent_coef*log_probs_fake).detach()
        else:
            raise NotImplementedError

        loss = loss_real + loss_fake

        # discriminator loss
        # reward is shaped
        gp_loss = self.gp_regularizer(disc_inp_real, disc_inp_fake)
        loss += gp_loss
        self.optimizer_discrim.zero_grad()
        loss.backward()
        self.optimizer_discrim.step()

        disc_stats = {
            'disc/loss': loss.item(),
            'disc/den_f/min': x_fake.min().item(),
            'disc/den_f/max': x_fake.max().item(),
            'disc/den_logpi/min': (ent_coef*log_probs_fake).min().item(),
            'disc/den_logpi/max': (ent_coef*log_probs_fake).max().item(),
            'disc/f_offset': self.discrim_net.reward_offset.item(),
            'disc/logits_real/min': logits_real.min().item(),
            'disc/logits_real/max': logits_real.max().item(),
            'disc/logits_fake/min': logits_fake.min().item(),
            'disc/logits_fake/max': logits_fake.max().item(),
        }

        post_stats = {}
        for _ in range(self.p_step):
            post_stats = self.update_posterior(batch, total_step=total_step)

        return {**disc_stats, **post_stats}
    
    def update_posterior(self, batch, total_step):

        s_sample = torch.FloatTensor(np.stack(batch.state)).to(device)
        a_sample = torch.FloatTensor(np.stack(batch.action)).to(device)
        ns_sample = torch.FloatTensor(np.stack(batch.next_state)).to(device)
        z_sample = torch.FloatTensor(np.stack(batch.latent_code)).view(-1, self.encode_dim).to(device)

        # infer na_sample - used for decoder input for l2scaler for regularization
        with torch.no_grad():
            na_sample = self.policy_updater.policy_net.greedy_action(torch.cat([ns_sample, z_sample], 1)).reshape(-1, self.action_dim)

        # embedder constraint distance computation
        if self.dl_type == 'l2':
            limit_dist_ = torch.sum(torch.square(ns_sample - s_sample), dim=1, keepdim=True)
        elif 'l2asymdist' in self.dl_type:
            if 'posx' in self.dl_type:
                crit_s = torch.greater(s_sample[:, 0], 0)
                crit_ns = torch.greater(ns_sample[:, 0], 0)
            elif 'negx' in self.dl_type:
                crit_s = torch.less(s_sample[:, 0], 0)
                crit_ns = torch.less(ns_sample[:, 0], 0)
            else:
                raise NotImplementedError
            if 'incr' in self.dl_type:
                crit_disp = torch.greater(ns_sample[:, 0], s_sample[:, 0])
            elif 'decr' in self.dl_type:
                crit_disp = torch.less(ns_sample[:, 0], s_sample[:, 0])
            else:
                raise NotImplementedError
            crit_bool = crit_s * crit_ns * crit_disp

            limit_dist_l2 = torch.sum(torch.square(ns_sample - s_sample), dim=1, keepdim=True)
            # expand, otherwise torch.where will broadcast wrong
            crit_bool = torch.unsqueeze(crit_bool, 1)
            limit_dist_ = torch.where(crit_bool, limit_dist_l2*0.0, limit_dist_l2)
        elif self.dl_type == 'disc':
            # this should be positive, use clip_discriminator arg
            # reward is not shaped
            if not self.cond_rew:
                # disc_inp_sample = torch.cat([s_sample, a_sample, ns_sample], 1)
                disc_inp_sample = torch.cat([s_sample, a_sample], 1)
                limit_dist_ = self.discrim_net(disc_inp_sample, pure_reward=True).detach()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # scale limit by l2_dist
        if self.dl_l2m == 0:
            # no scaling
            l2_dist_ = 1
        else:
            # calculate l2_dist_
            if not self.ac_dec:
                tp_sample = s_sample
                ntp_sample = ns_sample
            else:
                tp_sample = torch.cat([s_sample, a_sample], 1)
                ntp_sample = torch.cat([ns_sample, na_sample], 1)
            l2_dist_ = torch.linalg.norm(ntp_sample - tp_sample, dim=1, keepdim=True)
        # calc limit_dist
        limit_dist = limit_dist_/max(self.discrim_net.clip, 1e-3) * l2_dist_ * self.dl_scale

        # code pred loss
        code_loss = -torch.mean(self.decoder.get_logprob(s_sample, a_sample, z_sample))
        # dist limit loss
        embed_disp = self.decoder.get_code(ns_sample, na_sample) - self.decoder.get_code(s_sample, a_sample)
        embed_dist = torch.sum(torch.square(embed_disp), dim=1, keepdim=True)
        delta_dist = torch.square(limit_dist) - embed_dist

        violations = torch.minimum(delta_dist, self.slack_dual)
        violations_mean = torch.mean(violations)
        limit_loss = -torch.exp(self.loglambda_dual).detach() * violations_mean
        lambda_loss = torch.exp(self.loglambda_dual) * violations_mean.detach()

        loss = code_loss
        if self.regularize_decoder:
            loss += limit_loss

        self.optimizer_posterior.zero_grad()
        loss.backward()
        embed_gradnorm = sum([p.grad.detach().norm().item()**2 for p in self.decoder.parameters()])
        self.optimizer_posterior.step()

        self.optimizer_dual.zero_grad()
        lambda_loss.backward()
        self.optimizer_dual.step()

        # reinitialize zdemos
        self.reinitialize_misc()

        return {
            'post/loss': loss.item(),
            'post/code_loss': code_loss.item(),
            'post/lambda_loss': lambda_loss.item(),
            'post/violations/mean': violations_mean.item(),
            'post/lambda': torch.exp(self.loglambda_dual).item(),
            'post/embed_gradnorm': embed_gradnorm,
            'post/limit_dist/min': limit_dist.min().item(),
            'post/limit_dist/max': limit_dist.max().item(),
            'post/embed_dist/min': embed_dist.min().item(),
            'post/embed_dist/max': embed_dist.max().item(),
            'post/delta_dist/min': delta_dist.min().item(),
            'post/delta_dist/mean': delta_dist.mean().item(),
            'post/delta_dist/max': delta_dist.max().item(),
            'post/disc_reward_frac': self.disc_reward_frac,
            'post/post_reward_frac': self.post_reward_frac,
        }

    ## copied as is from InfoGAIL after cont dim adjustment
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
                # loss = 0.5 * ((action_mean - a_batch) ** 2 ).mean()    ##
                log_probs = policy_net.get_log_prob(s_batch, a_batch)
                loss = -log_probs.mean()

                optimizer_pi_bc.zero_grad()
                loss.backward()
                optimizer_pi_bc.step()

        t1 = time.time()
        print("Pi BC %s steps (%2.2fs). Final MSE loss %2.5f" % (colored(count, p_color), t1-t0, loss.item()))
