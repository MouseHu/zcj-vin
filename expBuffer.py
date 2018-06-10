import torch 
import torch.nn as nn
import numpy as np
class ExperienceBuffer(object):
	def __init__(self,config,maxexp=10000,cuda=True):
		self.pointer=0
		self.cuda=cuda
		self.maxexp=maxexp
		self.batch_size=config.batch_size
		self.config=config
		self.action=torch.Tensor(maxexp)
		self.state=torch.Tensor(maxexp,2,config.imsize,config.imsize)
		self.S1=torch.Tensor(maxexp)
		self.S2=torch.Tensor(maxexp)
		self.next_state=torch.Tensor(maxexp,2,config.imsize,config.imsize)
		self.next_S1=torch.Tensor(maxexp)
		self.next_S2=torch.Tensor(maxexp)
		self.reward=torch.Tensor(maxexp)
		self.over=torch.Tensor(maxexp)
		self.full=False
		if cuda:
			self.S1=self.S1.cuda()
			self.S2=self.S2.cuda()
			self.state=self.state.cuda()
			self.next_S1=self.next_S1.cuda()
			self.next_S2=self.next_S2.cuda()
			self.next_state=self.next_state.cuda()
			self.action=self.action.cuda()
	#(action,state,place,next_state,next_place,reward,over)
	def add(self,exp):
		self.pointer=(self.pointer+1)%self.maxexp
		self.action[self.pointer]=exp[0]
		self.state[self.pointer]=exp[1]
		self.next_state[self.pointer]=exp[3]
		self.S1[self.pointer]=exp[2][0]
		self.S2[self.pointer]=exp[2][1]
		self.next_S1[self.pointer]=exp[4][0]
		self.next_S2[self.pointer]=exp[4][1]
		self.reward[self.pointer]=exp[5]
		if exp[6]==True:
			self.over[self.pointer]=0
		else:
			self.over[self.pointer]=1
		if self.pointer==0:
			self.full=True
	def can_sample(self,length):
		if length>self.maxexp:
			return False
		if self.full:
			return True
		if self.pointer>length:
			return True
		return False
	def sample(self,length=0):
		if length==0:
			length=self.batch_size
		if self.can_sample==False:
			print("no!")
		
		self.s_action=torch.Tensor(length)
		self.s_state=torch.Tensor(length,2,self.config.imsize,self.config.imsize)
		self.s_S1=torch.Tensor(length)
		self.s_S2=torch.Tensor(length)
		self.s_next_state=torch.Tensor(length,2,self.config.imsize,self.config.imsize)
		self.s_next_S1=torch.Tensor(length)
		self.s_next_S2=torch.Tensor(length)
		self.s_reward=torch.Tensor(length)
		if self.cuda:
			self.s_action=self.s_action.cuda()
			self.s_state=self.s_state.cuda()
			self.s_S1=self.s_S1.cuda()
			self.s_S2=self.s_S2.cuda()
			self.s_next_state=self.s_next_state.cuda()
			self.s_next_S1=self.s_next_S1.cuda()
			self.s_next_S2=self.s_next_S2.cuda()
			self.s_reward=self.s_reward.cuda()
			#self.s_over=torch.Tensor(length)
		if self.full:
			maxexp=self.maxexp
		else:
			maxexp=self.pointer
		for i in range(length):
			x=np.random.randint(maxexp)
			while self.over[x]==1:
				x=np.random.randint(maxexp)
			self.s_action[i]=self.action[x]
			self.s_state[i]=self.state[x]
			self.s_S1[i]=self.S1[x]
			self.s_S2[i]=self.S2[x]
			self.s_next_state[i]=self.next_state[x]
			self.s_next_S1[i]=self.next_S1[x]
			self.s_next_S2[i]=self.next_S2[x]
			self.s_reward[i]=self.reward[x]
		return self.s_action,self.s_state,self.s_S1,self.s_S2,self.s_next_state,self.s_next_S1,self.s_next_S2,self.s_reward
		
