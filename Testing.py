import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.utils.net.common import Net   # you can define other net by following the API: https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network

from SnakeEnv import SnakeEnv

# Hyperparameters
lr = 1e-3                                    # learning rate
gamma = 0.95                                 # reward discount factor (higher means more focus on future rewards)
n_step = 3                                   # n-step TD 
target_freq = 320                            # network update frequency (how often to update the target network)


env = SnakeEnv(render_mode=True)

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[64, 128, 64])

optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)

policy.load_state_dict(torch.load('x.pth'))

policy.eval()
policy.set_eps(0)

env = ts.env.DummyVectorEnv([lambda: env])
collector = ts.data.Collector(policy, env)
collector.collect(n_episode=10, render=0.001)