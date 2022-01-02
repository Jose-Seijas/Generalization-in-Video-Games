
# Here's a bit of code that should help you get started on your projects.

# The cell below installs `procgen` and downloads a small `utils.py` script that contains some utility functions. You may want to inspect the file for more details.

###!pip install procgen
###!wget https://raw.githubusercontent.com/nicklashansen/ppo-procgen-utils/main/utils.py

# Hyperparameters. These values should be a good starting point. You can modify them later once you have a working implementation.

# Hyperparameters
total_steps = 16e6
num_envs = 32
num_levels = 10
num_steps = 512
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
feature_dim = 256
anneal_lr = True
lr_rate = 5e-4
frame_stack = 1

details={'total_steps': total_steps,
		'num_envs' : num_envs,
        'num_levels' : num_levels,
        'num_steps' : num_steps,
        'num_epochs' : num_epochs,
        'batch_size' : batch_size,
        'eps' : eps,
        'grad_eps' : grad_eps,
        'value_coef' : value_coef,
        'entropy_coef' : entropy_coef,
        'anneal_lr' : anneal_lr,
        'lr_rate' : lr_rate,
        'frame_stack' : frame_stack
			} 

# Network definitions. We have defined a policy network for you in advance. It uses the popular `NatureDQN` encoder architecture (see below), while policy and value 
# functions are linear projections from the encodings. There is plenty of opportunity to experiment with architectures, so feel free to do that! Perhaps implement the 
# `Impala` encoder from [this paper](https://arxiv.org/pdf/1802.01561.pdf) (perhaps minus the LSTM).

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils2 import make_env, Storage, orthogonal_init
import datetime
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
		self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
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
		return (self._out_channels, (h+1) // 2, (w + 1) // 2)

class Scale(nn.Module):
	def __init__(self, scale):
		super().__init__()
		self.scale = scale
	
	def forward(self, x):
		return x * self.scale

#class Encoder(nn.Module):
#  def __init__(self, in_channels, feature_dim):
#    super().__init__()
#    self.layers = nn.Sequential(
#        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
#        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
#        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
#        Flatten(),
#        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
#    )
#    self.apply(orthogonal_init)
#
#  def forward(self, x):
#    return self.layers(x)


class Policy(nn.Module):
	def __init__(self, n_channels, feature_dim, num_actions, flatten_size=32*8*8):
		super().__init__()
		shape = (n_channels, 64, 64)
		conv_seqs = [Scale(1/255)]
		for out_channels in [16, 32, 32]:
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
env = make_env(num_envs, num_levels=num_levels, env_name='fruitbot', fr_stack=frame_stack)
eval_env = make_env(num_envs, num_levels=num_levels, env_name='fruitbot', fr_stack=frame_stack)

# Define network
#encoder =  Encoder(3, feature_dim)
policy =  Policy(frame_stack*3, feature_dim,  env.action_space.n)
policy.cuda()

# Define optimizer
# these are reasonable values but probably not optimal
optimizer = torch.optim.Adam(policy.parameters(), lr_rate, eps=1e-5)
if anneal_lr:
	# https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
	lr = lambda f: f * lr_rate

fecha=datetime.datetime.now()
nombre = 'expRI2'+str(fecha.month)+'_'+ str(fecha.day)+'_'+str(fecha.hour)+str(fecha.minute)
with open(nombre+'.txt','w') as wf:
    for key, value in details.items():  
        wf.write('%s:%s\n' % (key, value))
    wf.write(str(policy.encoder)+'\n')
    wf.write(str(policy.policy)+'\n')
    wf.write(str(policy.value)+'\n')
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
update = 1
num_updates = int(total_steps/(num_envs*num_steps))
with open("results_trainingRI2.txt", "w") as f:
	while step < total_steps:
		# Define environment
		# check the utils.py file for info on arguments
		env = make_env(num_envs, num_levels=num_levels, env_name='fruitbot', fr_stack=frame_stack, seed = random.randint(0,2**31-1))
		###storage.reset()
		
		if update==int(num_updates/2):#update==int(num_updates/4) or update==int(num_updates/4*3)
			ff = open('model'+fnomb+'.pkl','wb')
			pickle.dump(policy, ff, pickle.HIGHEST_PROTOCOL)
			ff.close()
		
		# Annealing the rate if instructed to do so.
		if anneal_lr:
			frac = 1.0 - (update - 1.0) / (10*num_steps)
			lrnow = lr(frac)
			optimizer.param_groups[0]['lr'] = lrnow
		
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

		if aux == 25:
			# Make evaluation environment
			obs = eval_env.reset()

			frames = []
			total_reward = []

			# Evaluate policy
			policy.eval()
			###storage_test.reset()
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
torch.save(policy.state_dict, 'checkpointRI2.pt')

import pickle

with open ('modelRI2.pkl', 'wb') as f2:
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
imageio.mimsave('vid512RI2.gif', frames, fps=25)

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
imageio.mimsave('vid1024RI2.gif', frames, fps=25)
