import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import gym
import numpy as np
from video_recorder import VideoRecorder
import envpool

import random
from typing import Optional
import os
import pygame

import cv2
from mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar,num_procs,setup_mpi_gpus
from mpi4py import MPI

import faulthandler

faulthandler.enable()

BLACK = (0,0,0) #background 

DARK_GREEN = (0,100,0)#Snake head color
GREEN = (0, 255, 0) #Snake body color

RED = (255, 0, 0) #Fruit color
BROWN = (156,102,31) #Grid color


#Action masking
class CategoricalMasked(Categorical):
	def __init__(self, logits, mask: Optional[torch.Tensor] = None):
		self.mask = mask
		self.sz = mask.size(dim=0)
		if self.mask is None:
			super(CategoricalMasked, self).__init__(logits=logits)
		else:
			num_envs,num_actions = logits.size()
			self.boolean_mask  = torch.ones((self.sz,num_actions),dtype=torch.bool,device=logits.device)
			for i in range(0,self.sz):
				for j in range(0,num_actions):
					if j == self.mask[i]:
						self.boolean_mask[i][j] = False
			self.mask_value = torch.tensor(torch.finfo(logits.dtype).min,dtype=logits.dtype ,device=logits.device)
			self.logits = torch.where(self.boolean_mask, logits, self.mask_value)
			super(CategoricalMasked, self).__init__(logits=self.logits)
	def entropy(self):
		if self.mask is None:
			return super().entropy()
		p_log_p = self.probs*self.logits
		p_log_p = torch.where(self.boolean_mask,p_log_p,torch.tensor(0, dtype=p_log_p.dtype, device=p_log_p.device))
		return -torch.sum(p_log_p, dim= 1)



def layer_init(m,std=np.sqrt(2)):
	#print("within init_weights_and_biases")
	nn.init.orthogonal_(m.weight,std)
	nn.init.constant_(m.bias.data,0)
	return m


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
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class MLPActorCritic(nn.Module):
	def __init__(self):
		super(MLPActorCritic, self).__init__()
		shape = (1, 84, 84)
		conv_seqs = []
		for out_channels in [16, 32, 32]:
			conv_seq = ConvSequence(shape, out_channels)
			shape = conv_seq.get_output_shape()
			conv_seqs.append(conv_seq)
		conv_seqs += [
			nn.Flatten(),
			nn.ReLU(),
			nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256),
			nn.ReLU(),
		]
		self.network = nn.Sequential(*conv_seqs)
		self.actor = layer_init(nn.Linear(256, 4), std=0.01)
		self.critic = layer_init(nn.Linear(256, 1), std=1)

	def step(self,obs,masked_action,a=None,grad_condition=False):
		with torch.set_grad_enabled(grad_condition):
			logits = self.actor(self.network(obs.permute((0, 3, 1, 2)) / 255.0))
			pi = CategoricalMasked(logits,masked_action)
			if a == None:
				a = pi.sample()
				logp_a = pi.log_prob(a)
			else:
				logp_a = pi.log_prob(a)
			v = torch.squeeze(self.critic(self.network(obs.permute((0, 3, 1, 2)) / 255.0)),-1)
		return a,v,logp_a,pi.entropy()


class PPO: 
	def __init__(self):
		self.num_envs = 16 #previously32
		self.num_updates = 5000#previously5000 
		self.num_timesteps = 128 #previously128
		self.gamma = 0.99
		self.lamda = 0.95
		self.mini_batch_size = 512 #previously512
		self.learning_rate = 0.0002#previously0.0002
		self.clip_coef = 0.2
		self.entropy_coef=0.01  #previously0.01
		self.value_coef= 0.5
		self.max_grad_norm =0.5
		self.update = 0
		self.epochs = 4#previously4
		self.episode_return = 0.0
		self.episode_length = 0


	def capped_cubic_video_schedule(self) -> bool:
		#return True
		if(self.update>=4500):
			return True
		else:
			return False


	def render(self,fruit_position,head_position,body_positions,display_size=84,scale=2.1,body_width=8.4):
		surf = pygame.Surface((display_size,display_size))
		surf.fill(BLACK)
		pygame.transform.scale(surf, (scale,scale))

		pygame.draw.rect(surf,BROWN,pygame.Rect(0,0,display_size,display_size),int(body_width))

		if fruit_position[0] !=0 and fruit_position[1] != 0:
			pygame.draw.rect(surf,RED,pygame.Rect(fruit_position[0]*scale-body_width/2,fruit_position[1]*scale-body_width/2,body_width,body_width))

		if head_position[0] !=0 and head_position[1] != 0:
			pygame.draw.rect(surf,DARK_GREEN,pygame.Rect(head_position[0]*scale-body_width/2,head_position[1]*scale-body_width/2,body_width,body_width))  

		for i in range(0,len(body_positions)):
			if body_positions[i][0] != 0 and body_positions[i][1] != 0 :
				pygame.draw.rect(surf,GREEN,pygame.Rect(body_positions[i][0]*scale-body_width/2,body_positions[i][1]*scale-body_width/2,body_width,body_width))  

		temp_array = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

		if display_size==84:
			temp_array = cv2.cvtColor(temp_array, cv2.COLOR_RGB2GRAY)
			temp_array = np.expand_dims(temp_array, -1)

		return temp_array
		

	def calculate_gae(self,last_values,last_dones):
		next_nonterminal = None
		last_gae_lam = 0

		for t in reversed(range(self.num_timesteps)):
			if t == self.num_timesteps - 1:
				next_nonterminal = 1.0-last_dones
				next_values = last_values
			else:
				next_nonterminal = 1.0-self.batch_dones[t+1]
				next_values = self.batch_values[t+1]
	
			delta = self.batch_rewards[t]+self.gamma*next_nonterminal*next_values-self.batch_values[t] 			
			self.batch_advantages[t] = last_gae_lam = delta +self.gamma*next_nonterminal*self.lamda*last_gae_lam

		self.batch_returns = self.batch_advantages+self.batch_values



	def step(self):
		#print("Step function enter:")	

		self.batch_actions = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_values = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_logprobs_ac = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_entropies_agent = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_obs = torch.zeros(self.num_timesteps,self.num_envs,84,84,1).to(self.device)
		self.batch_rewards = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_dones = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)
		self.batch_masked_directions = torch.zeros(self.num_timesteps,self.num_envs).to(self.device)


		for i in range(0,self.num_timesteps):

			self.batch_obs[i] = self.next_obs
			self.batch_dones[i] = torch.from_numpy(self.next_dones).type(torch.float32).to(self.device)

			#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)
			#print(self.next_obs.shape)

			actions,values,logprobs_ac,entropies_agent = self.actor_critic.step(
				torch.as_tensor(self.next_obs,dtype = torch.float32).to(self.device),self.masked_directions_tensor)
			
			next_obs_tmp,rewards,self.next_dones,infos = self.snake_game_envs.step(actions.cpu().numpy())


			
			for j in range(0,self.num_envs):				
				self.next_obs[j] = torch.as_tensor(self.render(next_obs_tmp["fruit_position"][j],next_obs_tmp["head_position"][j],next_obs_tmp["body_positions"][j]),dtype=torch.float32).to(self.device)
				self.masked_directions_tensor[j] = torch.as_tensor(next_obs_tmp["masked_direction"][j],dtype =torch.float32).to(self.device)


			if(proc_id()==0):
				self.episode_return += rewards[0]
				self.episode_length += 1
				if(self.next_dones[0]):
					filestr = "Episode"+str(self.episodeid)+":"+"return="+str(round(self.episode_return,2))+",length="+str(self.episode_length)
					self.trainf.write(str(filestr)+"\n")
					self.episodeid +=1
					self.episode_return = 0.0
					self.episode_length = 0	
				if(self.capped_cubic_video_schedule()):
					display_size = 400
					scale = 10
					body_width =40
					frame = self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["head_position"][0],next_obs_tmp["body_positions"][0],display_size,scale,body_width)
					self.vi_rec.capture_frame(frame)
					if(self.next_dones[0]):
						self.vi_rec.close()
						video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"
						self.vi_rec = VideoRecorder(video_path)

			#print(actions.shape)
			#print(self.batch_actions.shape)	
			self.batch_actions[i] = actions
			self.batch_values[i] = values
			
			self.batch_logprobs_ac[i] = logprobs_ac

			#print("************")	
			#print(entropies_agent.size())
			self.batch_entropies_agent[i] = entropies_agent
			self.batch_rewards[i] = torch.from_numpy(rewards).type(torch.float32).to(self.device)

			self.batch_masked_directions[i] = self.masked_directions_tensor

		_,next_values,_,_ = self.actor_critic.step(torch.as_tensor(self.next_obs,dtype = torch.float32).to(self.device),self.masked_directions_tensor)

	

		self.batch_advantages = torch.zeros_like(self.batch_values).to(self.device)
		self.batch_returns = torch.zeros_like(self.batch_values).to(self.device)
		self.calculate_gae(next_values,torch.from_numpy(self.next_dones).type(torch.float32).to(self.device))	

		self.batch_size	= self.num_timesteps*self.num_envs

		self.batch_actions =  self.batch_actions.reshape(-1)
		self.batch_values = self.batch_values.reshape(-1) 
		self.batch_logprobs_ac = self.batch_logprobs_ac.reshape(-1)
		self.batch_entropies_agent = self.batch_entropies_agent.reshape(-1) 
		self.batch_obs = self.batch_obs.reshape((-1,)+(84,84,1))
		self.batch_rewards = self.batch_rewards.reshape(-1)
		self.batch_dones = self.batch_dones.reshape(-1)
		self.batch_advantages = self.batch_advantages.reshape(-1) 
		self.batch_returns = self.batch_returns.reshape(-1) 
		self.batch_masked_directions = self.batch_masked_directions.reshape(-1)



	def train(self):
		setup_mpi_gpus()
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if(proc_id()==0):
			print("enter train")

		setup_pytorch_for_mpi()	

		seed = 1

		np.random.seed(seed)
		random.seed(seed)


		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.batch_size =  self.num_timesteps*self.num_envs

		self.snake_game_envs = envpool.make_gym('SnakeDiscrete-v2',num_envs=self.num_envs,rank=proc_id())
		
		self.episodeid = 1
		
		self.vi_rec = None

		
		if(proc_id()==0):
			print('************************self.device*************************')
			print(self.device)
			video_path = os.path.abspath(os.getcwd())+"/video/"+"Episode_"+str(self.episodeid)+".mp4"
			self.vi_rec = VideoRecorder(video_path)

		next_obs_tmp = self.snake_game_envs.reset()

		self.masked_directions_tensor = torch.from_numpy(np.zeros(self.num_envs)).type(torch.float32).to(self.device)
				
		self.next_obs = torch.zeros(self.num_envs,84,84,1).to(self.device)

		for j in range(0,self.num_envs):
			self.next_obs[j] =torch.as_tensor(self.render(next_obs_tmp["fruit_position"][j],next_obs_tmp["head_position"][j],next_obs_tmp["body_positions"][j]),dtype =torch.float32).to(self.device)
			self.masked_directions_tensor[j] = torch.as_tensor(next_obs_tmp["masked_direction"][j],dtype =torch.float32).to(self.device)


		self.next_dones = np.zeros(self.num_envs)

		torch.manual_seed(seed)	
		self.actor_critic = MLPActorCritic().to(self.device)
		sync_params(self.actor_critic)

		self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),self.learning_rate)
		lr_current = self.learning_rate	
	
		if(proc_id()==0):
			self.trainf = open('TrainLog.txt','a')
	
		for update in range(1,self.num_updates+1):

			self.update = update  #Used in video recording schedule

			if(proc_id()==0):
				print("***********************Update num**********************:%d"%update)
					
			frac = 1.0 - (update - 1.0) / self.num_updates
			lr_current = frac * self.learning_rate
		
			for group in self.optimizer.param_groups:
				group['lr'] = lr_current

			self.step() #step the environment and actor critic to get one batch of data
			self.batch_indices = [i for i in range(0,self.batch_size)]
			random.shuffle(self.batch_indices)
			self.compute_gradients_and_optimize()

		if(proc_id()==0):
			self.trainf.close()
			self.vi_rec.close()
		self.snake_game_envs.close()



	def compute_gradients_and_optimize(self):
		for epoch in range(self.epochs):
			#print("enter for")
			i = 0
			while (i < self.batch_size):
				#print("enter while")
				start = i
				end = i+ self.mini_batch_size
				
				slice =self.batch_indices[start:end]

				_,new_v,new_logp_a,entropy = self.actor_critic.step(self.batch_obs[slice],self.batch_masked_directions[slice],self.batch_actions[slice],grad_condition=True)

				mini_batch_advantages = self.batch_advantages[slice]
				mini_batch_advantages_mean = mini_batch_advantages.mean()
				mini_batch_advantages_std = mini_batch_advantages.std()
				mini_batch_advantages = (mini_batch_advantages - mini_batch_advantages_mean)/(mini_batch_advantages_std + 1e-8)
				

				logratio = new_logp_a-self.batch_logprobs_ac[slice]
				ratio = logratio.exp()
				
				ploss1 = -mini_batch_advantages*ratio
				ploss2 = -mini_batch_advantages* torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) 
				ploss = torch.max(ploss1,ploss2).mean()

				vloss1 = (new_v-self.batch_returns[slice])**2
				vloss2 = ((self.batch_values[slice]+torch.clamp(new_v-self.batch_values[slice],-self.clip_coef,self.clip_coef))-self.batch_returns[slice])**2
				vloss = 0.5*torch.max(vloss1,vloss2).mean()

				entropy_loss = entropy.mean()

				loss = ploss - self.entropy_coef*entropy_loss + self.value_coef*vloss  

				self.optimizer.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
				mpi_avg_grads(self.actor_critic)
				self.optimizer.step()

				i = i+self.mini_batch_size

	
	def test(self):
		if proc_id()==0:	
			test_count_max = 10
			test_count = 0

			snake_game_env_test = envpool.make_gym('SnakeDiscrete-v2',num_envs=1,rank=proc_id())

			next_obs_tmp = snake_game_env_test.reset()

			masked_directions_tensor = torch.from_numpy(np.zeros(1)).type(torch.float32).to(self.device)
			
			next_obs = torch.zeros(1,84,84,1).to(self.device)

			episode_return =0.0
			episode_length =0 
			episodeid=0

			display_size =400
			body_width=40
			scale=10

			video_path = os.path.abspath(os.getcwd())+"/video-test/"+"Test_Episode_"+str(episodeid)+".mp4"
			vi_rec = VideoRecorder(video_path) 
			frame=self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["head_position"][0],next_obs_tmp["body_positions"][0],display_size,scale,body_width)
			vi_rec.capture_frame(frame)

			testf = open('TestLog.txt','a')

			while(test_count<test_count_max):

				#print("----------------------------------TIMESTEP NO:%d---------------------------------------"%i)
				#print(self.next_obs.shape)				
				masked_directions_tensor[0] = torch.as_tensor(next_obs_tmp["masked_direction"],dtype =torch.float32).to(self.device)

				next_obs[0] = torch.as_tensor(self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["head_position"][0],next_obs_tmp["body_positions"][0]),dtype =torch.float32).to(self.device)


				actions,values,logprobs_ac,entropies_agent = self.actor_critic.step(
					torch.as_tensor(next_obs,dtype = torch.float32).to(self.device),masked_directions_tensor)
				
				next_obs_tmp,rewards,next_dones,infos = snake_game_env_test.step(actions.cpu().numpy())


				episode_return += rewards[0]
				episode_length += 1

				
				frame=self.render(next_obs_tmp["fruit_position"][0],next_obs_tmp["head_position"][0],next_obs_tmp["body_positions"][0],display_size,scale,body_width)
				vi_rec.capture_frame(frame)
				if(next_dones[0]):					
					filestr = "Test_Episode"+str(episodeid)+":"+"return="+str(round(episode_return,2))+",length="+str(episode_length)
					testf.write(str(filestr)+"\n")
					episodeid +=1
					vi_rec.close()
					video_path = os.path.abspath(os.getcwd())+"/video-test/"+"Test_Episode_"+str(episodeid)+".mp4"
					vi_rec = VideoRecorder(video_path)
					episode_return = 0.0
					episode_length = 0		
					test_count +=1

			testf.close()
			vi_rec.close()
			snake_game_env_test.close()	

if __name__ == '__main__':
	ppo_obj = PPO()
	ppo_obj.train()
	ppo_obj.test()