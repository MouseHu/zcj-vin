# %load traincuda-td-online2.py
import gridworld2 as gw
import myvindyn

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
    return np.random.randint(8)

def vinPolicy(status,place):
    if np.random.random()<e:
        action=np.random.randint(8)
        return action
    S1=torch.Tensor([place[0]])#.expand(1,1)
    S2=torch.Tensor([place[1]])#.expand(1,1)
    #print(torch.Tensor(status).shape)
    X=torch.Tensor(status).expand(1, len(status),status[0].shape[0],status[0].shape[1])
    config=myvindyn.Config()
    q1,q2=VINdyn(X,S1,S2,myvindyn.Config())
    q1=q1
    q2=q2
    #print(q1)
    #print(q2.shape)
    _,action=torch.max(q2,dim=1)
    action=int(action)    
    #print(action)
    assert 0<=action and action<9
    return action

def evaluate(env,policy,iters=5000,show=False):
    total_reward=0
    success=0.0
    time2=time.time()
    for i in range(iters):
        status,place,reward,over=env.reset()
        t=0
        Tmax=100
        while over==False and t<Tmax:
            action=policy(status,place)
            if iters%100==0 and show:
                print(action)
                env.plot()
            
            status,place,reward,over=env.step(action)
            
            t+=1
        total_reward+=env.total_reward+0.0
        if env.total_reward>Tmax*env.step_reward:
            success+=1
        if i%100==0:
            print(i)
    return total_reward/iters,success/iters,time.time()-time2


model=sys.argv[1]
VINdyn=myvindyn.VINdyn(myvindyn.Config())
VINdyn.load_state_dict(torch.load(model,map_location='cpu'))
grid=gw.GridWorld2_8dir(8,8,nobstacle=4,moving=True)
e=0
print(VINdyn)
print(evaluate(grid,vinPolicy,iters=1000))