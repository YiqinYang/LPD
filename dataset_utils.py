import collections
from typing import Optional

from sklearn import datasets

import d4rl
import gym
import numpy as np
from tqdm import tqdm
import utils
from flow import FlowDistributionWrapper, RealNVP, NormalizingFlowModel
import math

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])

import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):

        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] 
        
        self.subseq_len = 10
        self.hidden_num, self.stack_num = 50, 1
        flows = [RealNVP(action_dim * self.subseq_len, cond_dim=state_dim, hidden_dim=self.hidden_num) for _ in range(self.stack_num)]
        model = NormalizingFlowModel(action_dim * self.subseq_len, flows).to(device)
        self.flow_dist = FlowDistributionWrapper(model)

        folder_name = f'./antmaze_medium_diverse_flow_prior_result_gaussian/result_length_{str(self.subseq_len)}_{str(self.hidden_num)}hidden_{str(self.stack_num)}stack'
 
        model_id = 350000
        model.load_state_dict(torch.load('%s/%s_flow.pth' % (folder_name, model_id), map_location=device))
        print('flow model load done: ', model_id)

        action_z, dataset_obs, dataset_next_obs, dataset_terminal, dataset_done, dataset_accumulate_reward = [], [], [], [], [], []
        dataset_seq_obs, dataset_seq_act = [], []
        N = dataset['rewards'].shape[0]
        for data_id in range(N):
            # data_id = data_id + 998000 # 检查错误
            if data_id + (self.subseq_len -1) == N: # 处理数据集的末端, 我们希望最后一个 seq_done 是 [0, 1] 
                break
            
            seq_obs = dataset['observations'][data_id:data_id+self.subseq_len]
            next_obs = dataset['next_observations'][data_id+self.subseq_len - 1]
            # seq_obs 取不到 data_id+subseq_len 处的点, next_obs 取 data_id+subseq_len - 1 处的点
            seq_action = dataset['actions'][data_id:data_id+self.subseq_len]
            seq_done = dones_float[data_id:data_id+self.subseq_len]

            # ----------discount reward---------------
            seq_reward = dataset['rewards'][data_id:data_id+self.subseq_len]
            reward_accumulate = 0
            for seq_reward_i in range(len(seq_reward)):
                reward_accumulate += math.pow(0.99, seq_reward_i) * seq_reward[seq_reward_i]

            # 处理一条轨迹的末端: 我们想要的数据是 [0, 0, 0, 1], 前面的条件区分 [0, 0, 1, 0]; 后面的条件区分 [1, 1, 1, 1]
            if ((True in seq_done) and (seq_done[-1] == False)) or sum(seq_done) > 1:
                continue

            dataset_obs.append(seq_obs[0])
            dataset_next_obs.append(next_obs)
            dataset_accumulate_reward.append(reward_accumulate)
            dataset_terminal.append(dataset['terminals'][data_id+self.subseq_len - 1])
            dataset_done.append(seq_done[-1])
            dataset_seq_obs.append(seq_obs)
            dataset_seq_act.append(seq_action)

        dataset_obs = np.array(dataset_obs)
        dataset_next_obs = np.array(dataset_next_obs)
        dataset_accumulate_reward = np.array(dataset_accumulate_reward)
        dataset_terminal = np.array(dataset_terminal)
        dataset_done = np.array(dataset_done)
        dataset_seq_obs = torch.as_tensor(np.array(dataset_obs), dtype=torch.float32).to(device)
        dataset_seq_act = torch.as_tensor(np.array(dataset_seq_act), dtype=torch.float32).to(device).view(len(dataset_seq_act), action_dim * self.subseq_len)

        sequence_length = 40000
        sequence_number = 24
        dataset_seq_obs_2 = dataset_seq_obs[sequence_length*sequence_number:]
        dataset_seq_act_2 = dataset_seq_act[sequence_length*sequence_number:]
        dataset_output = []
        for i in range(sequence_number):
            print(i)
            encode_z = self.flow_dist.encode_z(dataset_seq_act[sequence_length*i:sequence_length*(i+1)].view(-1, action_dim*self.subseq_len),
                    dataset_seq_obs[sequence_length*i:sequence_length*(i+1)].view(-1, state_dim))
            dataset_output.append(encode_z.detach().cpu().numpy())
        output_2 = self.flow_dist.encode_z(dataset_seq_act_2.view(-1, action_dim*self.subseq_len), 
                    dataset_seq_obs_2.view(-1, state_dim)).detach().cpu().numpy()
        dataset_output.append(output_2)
        action_z = np.concatenate(dataset_output, axis=0)

        # np.save('z_0', action_z[:10000, 0])
        # np.save('z_1', action_z[:10000, 1])
        # np.save('z_2', action_z[:10000, 2])
        # np.save('z_3', action_z[:10000, 3])
        # np.save('z_4', action_z[:10000, 4])

        # action_z = np.tanh(action_z/10)

        print('obs: ', dataset_obs.shape, 'act_z: ', action_z.shape, 'rew: ', dataset_accumulate_reward.shape, 
                'terminal: ', dataset_terminal.shape, 'dones: ', dataset_done.shape, 'next_obs: ', dataset_next_obs.shape)
        
        super().__init__(dataset_obs.astype(np.float32),
                    actions=action_z.astype(np.float32),
                    rewards=dataset_accumulate_reward.astype(np.float32),
                    masks=1.0 - dataset_terminal.astype(np.float32),
                    dones_float=dataset_done.astype(np.float32),
                    next_observations=dataset_next_obs.astype(np.float32),
                    size=len(dataset_obs))


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
