import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from tianshou.utils.net.common import Net 

from SnakeEnv import SnakeEnv


# Hyperparameters
lr = 1e-3                                 # learning rate
epoch = 150                               # max training epochs
batch_size = 128                          # batch size
gamma = 0.95                              # reward discount factor (higher means more focus on future rewards)
n_step = 3                                # n-step TD (how many steps to look ahead)
target_freq = 320                         # network update frequency (how often to update the target network)
buffer_size = 20000                       # replay buffer size (memory size of past episodes)
train_num, test_num = 1000, 50            # number of training and testing environments
eps_train, eps_test = 0.1, 0.01              # epsilon-greedy policy (how often to choose a random action)
step_per_epoch = 150                      # max steps per epoch
step_per_collect = 10                     # 
reward_threshold = 5000                   # threshold for reward to be considered solved

logger = ts.utils.TensorboardLogger(SummaryWriter('log/dqn'))

train_envs = ts.env.DummyVectorEnv([lambda: SnakeEnv() for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: SnakeEnv() for _ in range(test_num)])


env = SnakeEnv()
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[64, 128, 64])

optim = torch.optim.Adam(net.parameters(), lr=lr)
policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True) 


result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
    test_num, batch_size, update_per_step=1/step_per_collect,
    train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
    test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
    stop_fn=lambda mean_rewards: mean_rewards >= reward_threshold,
    logger=logger)
print(f'Finished training! Use {result["duration"]}')


torch.save(policy.state_dict(), 'x.pth')