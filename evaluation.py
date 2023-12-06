from typing import Dict

import flax.linen as nn
import gym
import numpy as np
import torch
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

from flow import FlowDistributionWrapper, RealNVP, NormalizingFlowModel

def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int, flow_module) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    
    avarage_return = 0
    episode_num = 0
    action_dim = env.action_space.shape[0] 
    for kk in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            # agent: Learner
            action = agent.sample_actions(observation, temperature=0.0)
            obs = torch.as_tensor(np.array(observation).reshape(1,-1), dtype=torch.float32).view(1, -1).to(device)
            
            # --------------------------tanh z-------------------------------
            # action[np.where(action>=1)]=0.99999999999
            # action[np.where(action<=-1)]=-0.9999999999

            # arc_action_z = -(1/2)*np.log(1 - action) + 1/2*np.log(1 + action)
            # arc_action_z = arc_action_z * 10
            # action_z = torch.as_tensor(arc_action_z, dtype=torch.float32).view(1, -1).to(device)
            # --------------------------tanh z-------------------------------

            action_z = torch.as_tensor(action, dtype=torch.float32).view(1, -1).to(device)

            decode_action_seq = flow_module.decode_z(state=obs, z=action_z).cpu().detach().data.numpy().flatten()
            decision_number = len(decode_action_seq) / action_dim
            
            # ----- normal ------
            for i in range(int(decision_number)):
                decode_action = decode_action_seq[i*action_dim: (i+1)*action_dim]
                observation, reward, done, info = env.step(decode_action)
                avarage_return += reward
                if done:
                    break

        episode_num += 1
        for k in stats.keys():
            stats[k].append(info['episode'][k])

    avarage_return = avarage_return / num_episodes
    for k, v in stats.items():
        stats[k] = np.mean(v)

    stats['return'] = avarage_return
    return stats
