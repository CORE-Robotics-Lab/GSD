
from my_utils import *
from core_nn.nn_irl import Discriminator


class SNDecoder(nn.Module):
    def __init__(self, state_dim, action_dim, encode_dim,
                 hidden_size=(256, 256), activation='relu',
                 use_action=False,
                 specnorm=False, specnorm_k=1.0, squash=False,
                 xavier=False, ortho=False):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.encode_dim = encode_dim
        self.use_action = use_action
        self.squash = squash
        self.specnorm = specnorm
        self.specnorm_k = specnorm_k

        if self.specnorm:
            self.activation = F.relu
        else:
            if activation == 'tanh':
                self.activation = torch.tanh
            elif activation == 'relu':
                self.activation = F.relu
            elif activation == 'sigmoid':
                self.activation = F.sigmoid
            elif activation == "leakyrelu":
                self.activation = F.leaky_relu

        self.affine_layers = nn.ModuleList()
        last_dim = self.state_dim + (self.action_dim if self.use_action else 0)

        snn_rpm = nn.utils.parametrizations.spectral_norm if self.specnorm else lambda x: x
        for nh in hidden_size:
            self.affine_layers.append(snn_rpm(nn_init(nn.Linear(last_dim, nh), xavier=xavier, ortho=ortho)))
            last_dim = nh
        self.mean_head = snn_rpm(nn_init(nn.Linear(last_dim, encode_dim), xavier=xavier, ortho=ortho))
        # self.logstd_head = lambda x: torch.zeros((x.size(0), encode_dim))
        # self.logstd_param = nn.Parameter(torch.zeros((encode_dim)))
        # self.logstd_head = lambda x: torch.zeros((x.size(0), 1)) + self.logstd_param.reshape((1, -1))
        # self.logstd_head = nn_init(nn.Linear(last_dim, encode_dim), xavier=xavier, ortho=ortho)
        self.logstd_head = snn_rpm(nn_init(nn.Linear(last_dim, encode_dim), xavier=xavier, ortho=ortho))

    def _get_gaussian_params(self, states, actions):
        # optionally add action to input
        if self.use_action:
            x = torch.cat([states, actions], 1)
        else:
            x = states
        # apply SN magnitude
        if self.specnorm:
            x = x * self.specnorm_k
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        return self.mean_head(x), self.logstd_head(x)

    def get_code(self, states, actions):
        code = self._get_gaussian_params(states, actions)[0]
        if self.squash:
            code = torch.tanh(code)
        return code

    def get_logprob(self, states, actions, query):
        code_mu, code_logstd_ = self._get_gaussian_params(states, actions)
        code_logstd = torch.clamp(code_logstd_, -20, 2)
        # code_logstd = torch.clamp(code_logstd_, -1, 2)
        # code_logstd = torch.clamp(code_logstd_, 0, 2)

        if self.squash:
            # clip to avoid NaN
            eps = torch.finfo(query.dtype).eps
            gaussian_query = torch.atanh(query.clamp(min=-1.0 + eps, max=1.0 - eps))
        else:
            gaussian_query = query

        code_var = torch.exp(code_logstd) ** 2
        code_lprob = -((gaussian_query - code_mu) ** 2) / (2 * code_var) - code_logstd - math.log(math.sqrt(2 * math.pi))
        # dimensions of latent space are deemed independent
        code_lprob = torch.sum(code_lprob, dim=1, keepdim=True)

        if self.squash:
            # input is query, not inverse of tanh
            code_lprob -= torch.sum(torch.log(1 - query**2 + 1e-8), dim=1, keepdim=True)
        return code_lprob

class PairDiscriminator(Discriminator):
    def __init__(self, state_dim, action_dim, use_action_not_next_state=False,
                 shaped_reward=False, offset_reward=False, symmetric=False, gamma=1.0,
                 *args, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.symmetric = symmetric
        self.use_action_not_next_state = use_action_not_next_state
        if self.symmetric:
            if self.use_action_not_next_state:
                print('Invalid with symmetric=True')
            # function only takes in state dim
            input_dim = state_dim
        else:
            # function takes in state, next state dim
            input_dim = state_dim*2
            if self.use_action_not_next_state:
                input_dim = state_dim+action_dim
        super().__init__(input_dim, 0, *args, **kwargs)
        self.shaped_reward = shaped_reward
        self.offset_reward = offset_reward
        if self.shaped_reward:
            self.gamma = gamma
            self.potential_layers = nn.ModuleList()
            last_dim = state_dim
            for nh in [100, 100, 1]:
                self.potential_layers.append(nn.Linear(last_dim, nh))
                last_dim = nh
        _ro_init = -self.clip/2 if self.clip > 0 else 0
        if self.offset_reward:
            self.reward_offset = nn.Parameter(torch.ones(())*_ro_init)
        else:
            self.reward_offset = torch.ones(())*_ro_init

    # copied from irl.py
    def _forward_reward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.score_out(x) 
        if self.clip > 0:   # if clip, we use sigmoid to bound in (0, clip) (positive)
            x = torch.sigmoid(x) * self.clip 
        if self.clip < 0:   # tanh to bound in (clip, -clip)
            x = torch.tanh(x) * -self.clip 
        return x

    def _forward_potential(self, x):
        for potential_layer in self.potential_layers:
            x = self.activation(potential_layer(x))
        return x

    # used by gp regularizer and everything else
    # to avoid modifying forward in gp regularizer
    # we split and handle concated here
    def forward(self, x, pure_reward=False):
        # assert x.shape[1] == self.state_dim*2 + self.action_dim, NotImplemented
        assert x.shape[1] == self.state_dim + self.action_dim, 's, a, ns form not implemented'
        s = x[:, :self.state_dim]
        ac = x[:, self.state_dim:self.state_dim + self.action_dim]
        # ns = x[:, self.state_dim + self.action_dim:]
        if self.symmetric:
            raise NotImplementedError
            # r_s = self._forward_reward(s)
            # r_ns = self._forward_reward(ns)
            # r_ret = (r_s + r_ns)/2
        else:
            if self.use_action_not_next_state:
                input_vec = torch.cat([s, ac], 1)
            else:
                raise NotImplementedError
                # input_vec = torch.cat([s, ns], 1)
            r_ret = self._forward_reward(input_vec)
        # reward shaping is disabled for limit and disc reward calculation
        if not pure_reward:
            if self.shaped_reward:
                raise NotImplementedError
                # v_s = self._forward_potential(s)
                # v_ns = self._forward_potential(ns)
                # r_ret += self.gamma * v_ns - v_s
            if self.offset_reward:
                r_ret += self.reward_offset
        return r_ret

    # only a convenient interface to avoid breaking agents that produce heatmaps
    def get_reward(self, *args, **kwargs):
        inp = torch.cat(args, 1)
        return self.forward(inp, **kwargs)
