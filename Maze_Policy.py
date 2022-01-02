
# Here's a bit of code that should help you get started on your projects.

# The cell below installs `procgen` and downloads a small `utils.py` script that contains some utility functions. You may want to inspect the file for more details.

###!pip install procgen
###!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py

# Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation.

# Hyperparameters
total_steps = 16e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
feature_dim = 256

details={'total_steps': total_steps,
		'num_envs' : num_envs,
        'num_levels' : num_levels,
        'num_steps' : num_steps,
        'num_epochs' : num_epochs,
        'batch_size' : batch_size,
        'eps' : eps,
        'grad_eps' : grad_eps,
        'value_coef' : value_coef,
        'entropy_coef' : entropy_coef} 

# Network definitions. We have defined a policy network for you in advance. It uses the popular `NatureDQN` encoder architecture (see below), while policy and value 
# functions are linear projections from the encodings. There is plenty of opportunity to experiment with architectures, so feel free to do that! Perhaps implement the 
# `Impala` encoder from [this paper](https://arxiv.org/pdf/1802.01561.pdf) (perhaps minus the LSTM).

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import datetime
import random

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy1 = orthogonal_init(nn.Linear(feature_dim, 64), gain=.01)
    self.policy2 = orthogonal_init(nn.Linear(64, num_actions), gain=.01)
    self.value1 = orthogonal_init(nn.Linear(feature_dim, 64), gain=1.)
    self.value2 = orthogonal_init(nn.Linear(64, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu(), dist

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy1(x)
    logits = nn.ReLU()(logits)
    logits = self.policy2(logits)
    x = self.value1(x)
    x = nn.ReLU()(x)
    value = self.value2(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value


# Define environment
# check the utils.py file for info on arguments
env = make_env(num_envs, num_levels=num_levels, env_name='maze')
print('Observation space:', env.observation_space)
print('Action space:', env.action_space.n)
eval_env = make_env(num_envs, num_levels=num_levels, env_name='maze')

# Define network
encoder =  Encoder(3, feature_dim)
policy =  Policy(encoder, feature_dim,  env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

fecha=datetime.datetime.now()
nombre = 'expRPol_MZ'+str(fecha.month)+'_'+ str(fecha.day)+'_'+str(fecha.hour)+str(fecha.minute)
with open(nombre+'.txt','w') as wf:
    for key, value in details.items():  
        wf.write('%s:%s\n' % (key, value))
    wf.write(str(policy.encoder)+'\n')
    wf.write(str(policy.policy1)+'\n')
    wf.write(str(policy.policy2)+'\n')
    wf.write(str(policy.value1)+'\n')
    wf.write(str(policy.value2)+'\n')
    wf.write(str(optimizer))

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

storage_test = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

# Run training
obs = env.reset()
step = 0
aux = 0
with open("results_trainingRPol_MZ.txt", "w") as f:
	while step < total_steps:
		# Define environment
		# check the utils.py file for info on arguments
		env = make_env(num_envs, num_levels=num_levels, env_name='maze', seed = random.randint(0,2**31-1))
		storage.reset()
		
		aux += 1
		# Use policy to collect data for num_steps steps
		policy.eval()
		for _ in range(num_steps):
			# Use policy
			action, log_prob, value, dist = policy.act(obs)

			# Take step in environment
			next_obs, reward, done, info = env.step(action)

			# Store data
			storage.store(obs, action, reward, done, info, log_prob, value)
		
			# Update current observation
			obs = next_obs

		# Add the last observation to collected data
		_, _, value, dist = policy.act(obs)
		storage.store_last(obs, value)

		# Compute return and advantage
		storage.compute_return_advantage()
		
		#Standard Deviation
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
				surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage
				pi_loss =  - torch.min(surr1, surr2).mean()

				# Clipped value function objective
				value_loss = (b_returns - new_value).pow(2).mean()

				# Entropy loss
				entropy_loss = new_dist.entropy().mean()

				# Backpropagate losses
				loss = 0.5 * value_loss + pi_loss - entropy_coef * entropy_loss
				loss.backward()

				# Clip gradients
				torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

				# Update policy
				optimizer.step()
				optimizer.zero_grad()

		if aux == 50:
			# Make evaluation environment
			obs = eval_env.reset()

			frames = []
			total_reward = []

			# Evaluate policy
			policy.eval()
			storage_test.reset()
			for _ in range(num_steps):

				# Use policy
				action, log_prob, value, dist = policy.act(obs)
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
			
			# Calculate average return
			total_reward = torch.stack(total_reward).sum(0).mean(0)
			
			#Standard Deviation
			std_test = []
			for i in range(storage_test.num_steps):
				info = storage_test.info[i]
				std_test.append([d['reward'] for d in info])
			std_test = torch.Tensor(std_test)
			
			#print(f'Step: {step}\tMean reward: {storage.get_reward()}')
			#print('Average return:', total_reward)
			#print(std_test.sum(0))
			print(f'{step}\t{storage.get_reward()}\t{torch.std(std_train.sum(0))}\t{storage_test.get_reward()}\t{torch.std(std_test.sum(0))}', file=f)
			#print(f'{step}\t{storage.get_reward()}\t{total_reward}', file=f)
			aux = 0
		
		# Update stats
		step += num_envs * num_steps

print('Completed training!')
torch.save(policy.state_dict, 'checkpointRPol_MZ.pt')

import pickle

with open ('modelRPol_MZ.pkl', 'wb') as f2:
	pickle.dump(policy, f2, pickle.HIGHEST_PROTOCOL)

# Below cell can be used for policy evaluation and saves an episode to mp4 for you to view.

import imageio

# Make evaluation environment
obs = eval_env.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
for _ in range(512):

	# Use policy
	action, log_prob, value, dist = policy.act(obs)
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
imageio.mimsave('vid512RPol_MZ.gif', frames, fps=25)

# Make evaluation environment
obs = eval_env.reset()

frames = []
total_reward = []

for _ in range(800):

	# Use policy
	action, log_prob, value, dist = policy.act(obs)
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
imageio.mimsave('vid1024RPol_MZ.gif', frames, fps=25)
