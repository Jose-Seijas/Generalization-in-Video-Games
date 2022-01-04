
# Here's a bit of code that should help you get started on your projects.

# The cell below installs `procgen` and downloads a small `utils.py` script that contains some utility functions. You may want to inspect the file for more details.

###!pip install procgen
###!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py

# Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation.

# Hyperparameters
total_steps = 12e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
clip_coef = .2
grad_eps = .5
value_coef = .5
lr_rate = 4e-4
entropy_coef = .01
feature_dim = 256
anneal_lr = True
clip_vloss = True
frame_stack = 4
details={'total_steps': total_steps,
		'num_envs' : num_envs,
        'num_levels' : num_levels,
        'num_steps' : num_steps,
        'num_epochs' : num_epochs,
        'batch_size' : batch_size,
        'clip_coef' : clip_coef,
        'grad_eps' : grad_eps,
        'value_coef' : value_coef,
        'lr_rate' : lr_rate,
        'entropy_coef' : entropy_coef,
	    'feature_dim' : feature_dim,
        'anneal_lr' : anneal_lr,
        'frame_stack' : frame_stack,
        'clip_vloss' : clip_vloss
           } 

import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils3 import make_env, Storage, orthogonal_init
import pickle
import random

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs

class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        # print(self._input_shape)
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

class Policy(nn.Module):
  def __init__(self, n_channels, feature_dim, num_actions, flatten_size=64*8*8):
    super().__init__()  
    shape = (n_channels, 64, 64)
    conv_seqs = [Scale(1/255)]  
    for out_channels in [32, 48, 64]:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()            
            conv_seqs.append(conv_seq)
    conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(in_features=flatten_size, out_features=feature_dim),
            nn.ReLU(),
        ]
    self.encoder = nn.Sequential(*conv_seqs)
    self.policy = nn.Linear(in_features=feature_dim, out_features=num_actions)
    self.value = nn.Linear(in_features=feature_dim, out_features=1)
  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu(), dist

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels,env_name='starpilot',fr_stack=frame_stack)

eval_env = make_env(num_envs, num_levels=num_levels,env_name='starpilot',fr_stack=frame_stack)

# Define network
# encoder =  Encoder(3, 100)
# encoder =  ImpalaCNN(3,feature_dim)
# print(encoder)
policy =  Policy(frame_stack*3, feature_dim,env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=lr_rate, eps=1e-5)
if anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * lr_rate

fecha=datetime.datetime.now()
fnomb = str(fecha.month)+'_'+ str(fecha.day)+'_'+str(fecha.hour)+str(fecha.minute)
with open('exp'+fnomb+'.txt','w') as wf:
    for key, value in details.items():  
        wf.write('%s:%s\n' % (key, value))
    wf.write(str(policy)+'\n')
    wf.write(str(optimizer))

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma=0.99,
    lmbda=0.95
)

storage_test = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs,
    gamma=0.99,
    lmbda=0.95
)
# Run training
obs = env.reset()
step = 0
# aux2 = 0
# aux1 = 0
aux = 0

update=1
num_updates=int(total_steps/(num_envs*num_steps))
with open('res_train'+fnomb+'.txt','w') as f:

  while step < total_steps:
    env = make_env(num_envs, num_levels=num_levels,env_name='starpilot',fr_stack=frame_stack, seed=random.randint(0,2**31-1) )

    if update==int(num_updates/2):#update==int(num_updates/4) or  or update==int(num_updates/4*3)
      ff = open('model_int'+fnomb+'.pkl', 'wb')
      pickle.dump(policy, ff, pickle.HIGHEST_PROTOCOL)
      ff.close()

    # Annealing the rate if instructed to do so.   
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / (3*num_updates)
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    # aux2 += 1
    # aux1 += 1
    aux += 1

    # Use policy to collect data for num_steps steps
    policy.eval()
    for _ in range(num_steps):
      # Use policy
      action, log_prob, value, _ = policy.act(obs)
      
      # Take step in environment
      next_obs, reward, done, info = env.step(action)

      # Store data
      storage.store(obs, action, reward, done, info, log_prob, value)
      
      # Update current observation
      obs = next_obs

    # Add the last observation to collected data
    _, _, value, _ = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()
    
    #standard dev
    std_train = []
    for i in range(storage.num_steps):
      info = storage.info[i]
      std_train.append([d['reward'] for d in info])
    std_train = torch.Tensor(std_train)
    
    # Optimize policy
    policy.train()
    for epoch in range(num_epochs):

      # Iterate over batches of transitions
      generator = storage.get_generator(batch_size)
      for batch in generator:
        b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

        # Get current policy outputs
        new_dist, new_value = policy(b_obs)
        new_log_prob = new_dist.log_prob(b_action)

        # Clipped policy objective
        ratio = (new_log_prob - b_log_prob).exp()
        surr1 = ratio * b_advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_advantage

        pi_loss =  - torch.min(surr1, surr2).mean()

        # Value loss
        # new_values = new_values.view(-1)
        if clip_vloss:
            v_loss_unclipped = ((new_value - b_returns) ** 2)
            v_clipped = b_value + torch.clamp(new_value - b_value, - clip_coef, clip_coef)
            v_loss_clipped = (v_clipped - b_returns)**2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            value_loss = 0.5 * v_loss_max.mean()
        else:
            value_loss = 0.5 * ((new_value - b_returns) ** 2).mean()

        # value_loss = 0.5 *(b_returns - new_value).pow(2).mean()

        # Entropy loss
        entropy_loss = new_dist.entropy().mean()

        # Backpropagate losses
        loss =  value_loss*value_coef + pi_loss - entropy_coef * entropy_loss
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

        # Update policy
        optimizer.step()
        optimizer.zero_grad()

    # if aux2 ==25:
    if aux ==25:
			# Make evaluation environment
      obs = eval_env.reset()

      # frames = []
      total_reward = []

			# Evaluate policy
      policy.eval()
      storage_test.reset()
      for _ in range(num_steps):

        # Use policy
        _, log_prob, value, dist = policy.act(obs)
        action = dist.logits.argmax(dim=1)
    
        # Take step in environment
        next_obs, reward, done, info = eval_env.step(action)
        total_reward.append(torch.Tensor(reward))
        # Store data
        storage_test.store(obs, action, reward, done, info, log_prob, value)
        
        # Update current observation
        obs = next_obs

      # Add the last observation to collected data
      _, _, value, dist = policy.act(obs)
      storage_test.store_last(obs, value)

      # Compute return and advantage
      storage_test.compute_return_advantage()
      
      #standard dev
      std_test = []
      for i in range(storage_test.num_steps):
        info = storage_test.info[i]
        std_test.append([d['reward'] for d in info])
      std_test = torch.Tensor(std_test)

      # Calculate average return
      total_reward = torch.stack(total_reward).sum(0).mean(0)

      print(f'{step}\t{storage.get_reward()}\t{torch.std(std_train.sum(0) )}\t{storage_test.get_reward()}\t{torch.std(std_test.sum(0) )}', file=f)

      aux = 0

    step += num_envs * num_steps
    update+=1
  
print('Completed training!')
torch.save(policy.state_dict, 'checkpoint.pt')


with open ('model'+fnomb+'.pkl', 'wb') as f2:
	pickle.dump(policy, f2, pickle.HIGHEST_PROTOCOL)

# Below cell can be used for policy evaluation and saves an episode to mp4 for you to view.

import imageio

# Make evaluation environment
# eval_env = make_env(num_envs,  num_levels=num_levels, fr_stack=frame_stack)
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):#2056

  # Use policy
  _, _, _, dist = policy.act(obs)
  action = dist.logits.argmax(dim=1)
  # Take step in environment
  obs, reward, done, info = eval_env.step(action)
  total_reward.append(torch.Tensor(reward))

  # Render environment and store
  frame = (torch.Tensor(eval_env.render(mode='rgb_array'))*255.).byte()
  frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid'+fnomb+'.mp4', frames, fps=25)