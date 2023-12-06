from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import numpy as np
import math
import gym
import utils
import d4rl
import torch
import torch.nn as nn
from rlkit.torch.opal.utils import kld_gauss
# from utils import kld_gauss
from rlkit.torch import pytorch_util as ptu

class FCNetwork(nn.Module):
	def __init__(self, inp_dim, hidden_dims, out_dim, act_fn=nn.ReLU()):
		super(FCNetwork, self).__init__()
		self.inp_dim = inp_dim
		self.out_dim = out_dim
		self.hidden_dims = hidden_dims
		self.learn = True

		layer_lst = []
		in_dim = inp_dim

		for hidden_dim in hidden_dims:
			layer_lst.append(nn.Linear(in_dim, hidden_dim))
			layer_lst.append(act_fn)
			in_dim = hidden_dim

		layer_lst.append(nn.Linear(hidden_dim, out_dim))		#network output does not have activation function

		self.network = nn.Sequential(*layer_lst)

	def forward(self, inp):
		return self.network(inp)

	@property
	def num_layers(self):    	
		return len(self.hidden_dims)+1

class PolicyNetwork(nn.Module):
	def __init__(self, latent_dim, state_dim, hidden_dims, act_dim, masked_dim=0, act_fn=nn.ReLU()):
		super(PolicyNetwork, self).__init__()
		self.latent_dim = latent_dim
		self.state_dim = state_dim
		self.act_dim = act_dim
		self.masked_dim = masked_dim
		self.base = FCNetwork(inp_dim=latent_dim+state_dim-masked_dim, hidden_dims=hidden_dims, out_dim=2*act_dim, act_fn=act_fn)

	def forward(self, latent, state):
		if state is None:
			inp = latent
		elif latent is None:
			inp = state[:, self.masked_dim:]
		else:
			inp = torch.cat([latent, state[:, self.masked_dim:]], dim=1)
		base_out = self.base(inp)
		mean = base_out[:,:self.act_dim]
		log_std = base_out[:,self.act_dim:]
		std = log_std.exp()
		assert not torch.any(torch.isnan(base_out))
		return mean, std

	def act(self, latent, state, deterministic=False):
		mean, std = self.forward(latent, state)
		if deterministic:
			return mean
		else:
			act_dist = torch.distributions.Normal(mean, std)			# create a n-distributions around n-means with n-standard deviations
			return act_dist.sample()


	def calc_log_prob(self, latent, state, action):
		mean, std =  self.forward(latent, state)
		act_dist = torch.distributions.Normal(mean, std)
		log_prob = act_dist.log_prob(action).sum(-1)					# calculate the log_prob of the actions (of both those selected, and those not selected). Sum of log_prob in log space is equivalent to product of prob in real space
		return log_prob.mean()


class Gaussian:
    """ Represents a gaussian distribution """
    def __init__(self, mu, log_sigma=None):
        if log_sigma is None:
            if not isinstance(mu, torch.Tensor):
                import pdb; pdb.set_trace()
            mu, log_sigma = torch.chunk(mu, 2, -1)
            
        self.mu = mu
        self.log_sigma = torch.clamp(log_sigma, min=-10, max=2) if isinstance(log_sigma, torch.Tensor) else \
                            np.clip(log_sigma, a_min=-10, a_max=2)
        self._sigma = None
        
    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.sigma)

    def nll(self, x):
        # Negative log likelihood (probability)
        return -1 * self.log_prob(x)

    def log_prob(self, val):
        """Computes the log-probability of a value under the Gaussian distribution."""
        return -1 * ((val - self.mu) ** 2) / (2 * self.sigma**2) - self.log_sigma - math.log(math.sqrt(2*math.pi))

    @property
    def sigma(self):
        if self._sigma is None:
            self._sigma = self.log_sigma.exp()
        return self._sigma

    @property
    def shape(self):
        return self.mu.shape

    def __getitem__(self, item):
        return Gaussian(self.mu[item], self.log_sigma[item])
 
class MultivariateGaussian(Gaussian):
    def log_prob(self, val):
        return super().log_prob(val).sum(-1)

class NormalizingFlowModel(nn.Module):
    """
    Joins multiple flow models into composite flow.
    Implementation extended from: https://github.com/tonyduan/normalizing-flows/blob/master/nf/models.py
    """

    def __init__(self, flow_dim, flows):
        super().__init__()
        self._flow_dim = flow_dim
        # _flow_dim: 10; action_dim
        self.flows = nn.ModuleList(flows)
        # self.flows: RealNVP

    def forward(self, x, cond_inputs=None):
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device)
        for flow in self.flows: # flow: RealNVP
            x, ld = flow.forward(x, cond_inputs)
            log_det += ld

        z, prior_logprob = x, self._get_prior(m, x.device).log_prob(x)
        return z, prior_logprob, log_det

    def inverse(self, z, cond_inputs=None):
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device)
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z, cond_inputs)
            log_det += ld
        x = z
        return x, log_det

    def sample(self, num_samples=None, device=None, cond_inputs=None):
        if num_samples is None:
            num_samples = cond_inputs.shape[0]
        if device is None:
            device = cond_inputs.device
        z = self._get_prior(batch_size=num_samples, device=device).sample()
        x, _ = self.inverse(z, cond_inputs)
        return x

    def decode(self, z, cond_inputs):
        x, _ = self.inverse(z, cond_inputs)
        return x

    def _get_prior(self, batch_size, device):
        return MultivariateGaussian(torch.zeros((batch_size, self._flow_dim), requires_grad=False, device=device),
                                    torch.zeros((batch_size, self._flow_dim), requires_grad=False, device=device))


class RealNVP(nn.Module):
    """
    Non-volume preserving flow.
    [Dinh et. al. 2017]
    Implementation extended from: https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py
    """
    def __init__(self, dim, cond_dim=None, hidden_dim=100):
        """Constructs RealNVP flow. Note that input_dim == output_dim == dim.
           cond_dim allows to add conditioning to the flow model.
        """
        super().__init__()
        assert dim % 2 == 0 
        self.dim = dim # dim: (8,) action_dim
        self.cond_dim = cond_dim # cond_dim: (29,)
        input_dim = self.dim // 2 if cond_dim is None else self.dim // 2 + cond_dim
        # input_dim: (33,) action_dim / 2 + state_dim ; out_dim: (4,) action_dim / 2
        self.t1 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.s1 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.t2 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)
        self.s2 = FCNN(in_dim=input_dim, out_dim=dim // 2, hidden_dim=hidden_dim)

    def forward(self, x, cond_inputs=None):
        """Forward pass of the RealNVP module. Cond_inputs is a list of conditioning tensors."""
        assert len(x.shape) == 2 and x.shape[-1] == self.dim
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:] # lower: (128, 5); cond_inputs: (128, 128)

        t1_transformed = self.t1(lower, cond_inputs)
        s1_transformed = self.s1(lower, cond_inputs) 
        # t1_transformed: (128, 4); s1_transformed: (128, 4)

        upper = t1_transformed + upper * torch.exp(s1_transformed)
        
        t2_transformed = self.t2(upper, cond_inputs)
        s2_transformed = self.s2(upper, cond_inputs)
        
        lower = t2_transformed + lower * torch.exp(s2_transformed)
        z = torch.cat([lower, upper], dim=1)
        # z: (256, 8)
        log_det = torch.sum(s1_transformed, dim=1) + \
                  torch.sum(s2_transformed, dim=1)
        
        return z, log_det

    def inverse(self, z, cond_inputs=None):
        assert len(z.shape) == 2 and z.shape[-1] == self.dim
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        t2_transformed = self.t2(upper, cond_inputs)
        s2_transformed = self.s2(upper, cond_inputs)
        lower = (lower - t2_transformed) * torch.exp(-s2_transformed)
        
        t1_transformed = self.t1(lower, cond_inputs)
        s1_transformed = self.s1(lower, cond_inputs)
        upper = (upper - t1_transformed) * torch.exp(-s1_transformed)
        
        x = torch.cat([lower, upper], dim=1)
        log_det = torch.sum(-s1_transformed, dim=1) + \
                  torch.sum(-s2_transformed, dim=1)
        return x, log_det


class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        # in_dim: 133; out_dim: 5; hidden_dim: 32
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, additional_inputs):
        input = torch.cat([x] + [additional_inputs], dim=-1) if additional_inputs is not None else x
        return self.network(input)


class FlowDistributionWrapper:
    """Lightweight wrapper around flow model that makes it behave like distribution."""
    def __init__(self, flow):
        self._flow = flow
        # _flow: NormalizingFlowModel

    def log_prob(self, x, state):
        # x: (128, 10)
        u, prior_logprob, log_det = self._flow(x, state)
        # u: (128, 10)
        return u, log_det + prior_logprob

    def nll(self, x, state):
        # x: (128, 10)
        z, logp =  self.log_prob(x, state)
        return z, -logp

    def encode_z(self, x, state):
        u, prior_logprob, log_det = self._flow(x, state)
        return u

    def sample(self, state):
        # _flow: NormalizingFlowModel
        return self._flow.sample(cond_inputs=state)
 
    def decode_z(self, z, state):
        # action = self._flow.sample(cond_inputs=state)
        action = self._flow.decode(z, cond_inputs=state)
        return action

if __name__ == "__main__":
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    import torch.nn.functional as F

    def dataloader(env_name, subseq_len):
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)

        dones_float = np.zeros_like(dataset['rewards'])
        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] - dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        dones_float[-1] = 1

        seq_end_idxs = np.where(dones_float)[0]
        start, seqs = 0, []
        for end_idx in seq_end_idxs:
            num_in_traj = end_idx - start + 1
            if (num_in_traj <= subseq_len) or (num_in_traj <= 1):
                start = end_idx + 1
                continue
            seqs.append({'states': dataset['observations'][start:end_idx+1],
                'actions': dataset['actions'][start:end_idx+1],
                'rewards': dataset['rewards'][start:end_idx+1],
                'dones': dones_float[start:end_idx+1]
            })
            start = end_idx + 1
        return seqs

    def reload(seqs, subseq_len):
        start, traj_number = 0, len(seqs)
        output_state = []
        output_action = []
        for selected_seq_number in range(traj_number):
            print('traj: ', selected_seq_number, traj_number)
            # if selected_seq_number >= 5:
            #     break
            selected_seq = seqs[selected_seq_number]
            traj_length = selected_seq['states'].shape[0]
            for start_idx in range(traj_length - subseq_len + 1):
                output_state.append(
                    torch.FloatTensor(
                        selected_seq['states'][start_idx:start_idx+subseq_len]).to(device))
                output_action.append(
                    torch.FloatTensor(
                        selected_seq['actions'][start_idx:start_idx+subseq_len]).to(device))  
                output_done = selected_seq['dones'][start_idx:start_idx+subseq_len]

        output_state = torch.stack(output_state).to(device)
        output_action = torch.stack(output_action).to(device) 
        output = {'state': output_state, 'action': output_action}
        return output

    def sampler(seq):
        dataset_size = len(seq['state'])
        ind = np.random.randint(0, dataset_size, size=256)
        output_state = seq['state'][ind]
        output_action = seq['action'][ind]
        output = {'state': output_state, 'action': output_action}
        return output

    env_name = 'antmaze-medium-diverse-v0'
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])

    subseq_len = 10
    hidden_num = 50
    stack_num = 1
    folder_name = f'./antmaze_medium_diverse_flow_prior_result_gaussian/result_length_{str(subseq_len)}_{str(hidden_num)}hidden_{str(stack_num)}stack'
    seqs = dataloader(env_name, subseq_len)
    reload_seqs = reload(seqs, subseq_len)

    flows = [RealNVP(action_dim * subseq_len, cond_dim=state_dim, hidden_dim=hidden_num) for _ in range(stack_num)]
    model = NormalizingFlowModel(action_dim * subseq_len, flows).to(device)
    flow_dist = FlowDistributionWrapper(model)
    prior = PolicyNetwork(latent_dim=0, state_dim=state_dim, hidden_dims=[200, 200], act_dim=action_dim * subseq_len, act_fn=nn.ReLU()).to(device)
    model_parameters = list(model.parameters()) + list(prior.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=1e-4)

    # model.load_state_dict(torch.load('%s/%s_flow.pth' % (folder_name, 450000), map_location=device))
    print('flow model load done')

    for iter in range(500000):
        output = sampler(reload_seqs)
        input_state = output['state'][:, 0] 
        input_action = output['action'].view(256, action_dim * subseq_len)
        z, loss_flow = flow_dist.nll(input_action, input_state)
        loss_flow = loss_flow.mean()

        prior_sample = prior.act(latent=None, state=input_state)
        loss_prior = ((prior_sample - z)**2).mean()

        loss = loss_flow + 0.1 * loss_prior

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 50000 == 0:
            print(f"Iter: {iter}\t" +
                  f"NLL: {loss.mean().data:.2f}\t")
            torch.save(model.state_dict(), '%s/%s_flow.pth' % (folder_name, iter))