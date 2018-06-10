import gridworld2 as gw
import myvin

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import copy

import time
import sys
import itertools
def randomWalk(status,place):
    return np.random.randint(9)

def vinPolicy(status,place):
    if np.random.random()<e:
        action=np.random.randint(9)
        return action
    S1=torch.Tensor([place[0]]).cuda()
    S2=torch.Tensor([place[1]]).cuda()
    X=torch.Tensor(status).expand(1, len(status),status[0].shape[0],status[0].shape[1]).cuda()
    config=myvin.Config()
    q1,q2=VIN(X,S1,S2,myvin.Config())
    q1=q1.cuda()
    q2=q2.cuda()
    #print(q1)
    #print(q2.shape)
    _,action=torch.max(q1,dim=1)
    action=int(action)    
    #print(action)
    assert 0<=action and action<9
    return action
def evaluate(env,policy,iters=5000):
	total_reward=0
	time2=time.time()
	for i in range(iters):
		status,place,reward,over=env.reset()
		t=0
		while over==False and t<100:
			action=policy(status,place)
			status,place,reward,over=env.step(action)
			t+=1
		total_reward+=env.total_reward+0.0
		if i%100==0:
			print(i)
	return total_reward/iters,time.time()-time2
device=0
if len(sys.argv)>1:
   device=int(sys.argv[1])
with torch.cuda.device(device):
	
	VIN=myvin.VIN(myvin.Config()).cuda()
	VIN.load_state_dict(torch.load("model2/moving-model-9-3920.pkl"))#3920
	print(VIN)
	oldVIN=myvin.VIN(myvin.Config()).cuda()
	oldVIN.load_state_dict(VIN.state_dict())
	grid=gw.GridWorld2_8dir(8,8,nobstacle=4,moving=True)
	e=0
	#print(evaluate(grid,vinPolicy,1000))
	print(evaluate(grid,randomWalk))
	for i in range(10):
		print(evaluate(grid,vinPolicy,iters=1000))
		
		#print total_reward/iters,time.time()-time2
