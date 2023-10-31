from my_utils import *
from core_nn.nn_ac import *
from core.ac import PPO

""" PPO. With BC Annealing from Jena, et. al. """
class PPOBC(PPO): 
    def __init__(self, state_dim, action_dim, args, a_bound=1, is_discrete=False, encode_dim=0): 
        super().__init__(state_dim, action_dim, args, a_bound, is_discrete, encode_dim)

    def update_curriculum(self, progress):
        self.progress = np.clip(progress, 0, 1)

    def add_demos(self, du, args):
        # add pointer to disc_updater
        self.du = du
        # stored hyps
        self.il_method = args.il_method
        self.bc_cf = args.bc_cf
        self.bc_hl = args.bc_hl

        # init rate
        self.wt_zero = 0.5 ** (1/self.bc_hl)

    def _compute_bc_curr_wt(self):
        return self.wt_zero ** self.progress

    def _compute_bc_loss(self, batch_size):
        index = self.du.index_sampler(batch_size=batch_size)
        s_expert, a_expert = self.du.get_bc_data(index)
        log_probs = self.policy_net.get_log_prob(s_expert, a_expert)
        bc_loss = -log_probs.mean()
        return bc_loss

    def update_policy(self, states, actions, next_states, rewards, masks):
        with torch.no_grad():
            values = self.value_net.get_value(states).data
            fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        """ get GAE from trajectories """
        advantages, returns = self.estimate_advantages(rewards, masks, values)   # advantage is GAE and returns is the TD(lambda) return.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        fixed_log_probs = fixed_log_probs.detach().data

        target_kl = 0.01
        
        train = data_utils.TensorDataset(states, actions, returns, advantages, fixed_log_probs)
        if self.full_batch:
            self.train_epoch = 80
            train_loader = data_utils.DataLoader(train, batch_size=states.size(0), shuffle=True)
        else:
            train_loader = data_utils.DataLoader(train, batch_size=self.mini_batch_size, shuffle=True)


        for _ in range(self.train_epoch):
            for batch_idx, (states_batch, actions_batch, returns_batch, advantages_batch, fixed_log_probs_batch) in enumerate(train_loader):
                log_probs = self.policy_net.get_log_prob(states_batch, actions_batch)
                cur_entropy = self.policy_net.compute_entropy().mean()

                ratio = torch.exp(log_probs - fixed_log_probs_batch)                     
                min_adv = torch.where(advantages_batch>0, (1+self.ppo_clip)*advantages_batch, (1-self.ppo_clip)*advantages_batch)
                rl_loss = -torch.min(ratio * advantages_batch, min_adv).mean() - self.entropy_coef * cur_entropy

                # only modification done in this fn
                bc_curr_wt = self._compute_bc_curr_wt()
                bc_loss = self._compute_bc_loss(states_batch.shape[0]) * self.bc_cf
                policy_loss = (1-bc_curr_wt) * rl_loss + bc_curr_wt * bc_loss

                value_loss = (self.value_net.get_value(states_batch) - returns_batch).pow(2).mean()
                if self.gae_l2_reg > 0:
                    for params in self.value_net.parameters():
                        value_loss += self.gae_l2_reg * params.norm(2)

                if self.separate_net:
                    self.optimizer_policy.zero_grad() 
                    policy_loss.backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_policy.step()

                    self.optimizer_value.zero_grad() 
                    value_loss.backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.value_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_value.step()

                else:
                    self.optimizer_pv.zero_grad() 
                    (policy_loss + value_loss).backward() 
                    if self.ppo_gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.ppo_gradient_clip)
                    self.optimizer_pv.step()
                
                stats = {
                    'policy_loss': policy_loss.item(),
                    'value_loss': value_loss.item(),
                    'rl_loss': rl_loss.item(),
                    'bc_loss': bc_loss.item(),
                    'bc_curr_wt': bc_curr_wt,
                }

                ## early stopping based on KL 
                if self.ppo_early:
                    kl = (fixed_log_probs_batch - log_probs).mean()
                    if kl > 1.5 * target_kl:
                        # print('Early stopping at step %d due to reaching max kl.' % i)
                        if self.full_batch:
                            return stats
                        else:
                            break
        return stats
