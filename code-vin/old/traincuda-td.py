import gridworld as gw
import myvin

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import copy
import time
import sys
#import torchvision.transforms as transforms
grid=gw.GridWorld_8dir(nobstacle=5,moving=False)
#grid.show()
#grid.plot()
#for _ in range(100):
#    grid.step(grid.sample())
#grid.plot()
def randomWalk(status,place):
    return np.random.randint(9)
def evaluate(env,policy,iters=5000):
	    total_reward=0
	    for i in range(iters):
		status,place,reward,over,action=env.reset()
		while over==False:
		    status,place,reward,over,action=env.step(policy(status,place))
		total_reward+=env.total_reward+0.0
		if i%100==0:
		    print(i)
	    return total_reward/iters
device=0
if len(sys.argv)>1:
   device=int(sys.argv[1])
with torch.cuda.device(device):
	VIN=myvin.VIN(myvin.Config())
	VIN.cuda()
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
	epoches=2000
	episodes=200
	print("here")
	oldVIN=copy.deepcopy(VIN).cuda()
	#experience
	for k in range(epoches):
	    gamma=0.99
	    #step
	    replay=[]
	    #rewards=[]
	    print("begin")
	    time1=time.time()
	    for i in range(episodes):
		#reward=[]
		e=40.0/(40+k)#8.0/(10+k)# is not ok why?
		if i==0:
		    experience=grid.run_episode(vinPolicy)#,show=True)
		else:
		    experience=grid.run_episode(vinPolicy)
		#make discounted reward
		if i%100==0:
		    print(i)
		replay.append(experience)
	    #replay & update
	    #print(rewards)
	    print("experience",k,time.time()-time1)
	    time1=time.time()	
	    for i in range(500):
		X=[]
		S1=[]
		S2=[]
		oldS1=[]#next action
		oldS2=[]#next action
		oldX=[]
		action=[]
		Y=[]#torch.Tensor(reward[::-1])
		index=[]
		for j in range(myvin.Config().batch_size):# sample experience from replay
		    x1=np.random.randint(episodes)
		    x2=np.random.randint(len(replay[x1])-1)
		    #status,place,reward,over,action
		    
		    Y.append(replay[x1][x2][2])
		    action.append(replay[x1][x2][4])
		    X.append(replay[x1][x2][0])
		    oldX.append(replay[x1][x2+1][0])
		    S1.append(replay[x1][x2][1][0])
		    S2.append(replay[x1][x2][1][1])
		    oldS1.append(replay[x1][x2+1][1][0])
		    oldS2.append(replay[x1][x2+1][1][1])
		    index.append((x1,x2+1))

		X=torch.from_numpy(np.array(X)).float().cuda()#do not change it to torch.Tensor(X).float()
		S1=torch.from_numpy(np.array(S1)).float().cuda()
		S2=torch.from_numpy(np.array(S2)).float().cuda()
		oldS1=torch.from_numpy(np.array(oldS1)).float().cuda()
		oldS2=torch.from_numpy(np.array(oldS2)).float().cuda()
		oldX=torch.from_numpy(np.array(oldX)).float().cuda()
		action=torch.from_numpy(np.array(action)).unsqueeze(dim=1).long().cuda()
		Y=torch.from_numpy(np.array(Y)).float().cuda()
		#Qmax=torch.Tensor([replay[x[0]][x[1]][4] for x in index]).float() .cuda()


		oldoutputs, _ = oldVIN(oldX,oldS1,oldS2 ,  myvin.Config())
		Qmax=(torch.max(oldoutputs,dim=1)[0]).squeeze().cuda()
		outputs, _ = VIN(X,S1,S2 ,  myvin.Config())
		Qvalue=outputs.gather(index=action,dim=1).squeeze().cuda()
		#print(Qvalue.shape)
		#print(Y.shape)
		TDtarget=(Y+gamma*Qmax).cuda()
		criterion = torch.nn.MSELoss(size_average=False)
		loss=criterion(Qvalue,Y).cuda()
		optimizer = optim.RMSprop(VIN.parameters(), lr=2e-4, eps=1e-6)   
		loss.backward()
		# Update params
		optimizer.step()
		if i%100==0:
		    print(i)
	    print("update",k,time.time()-time1)
	    if k%100==99:	    
		print(evaluate(grid,vinPolicy))#its a bad policy :(
	#evaluate(grid,randomWalk)
	#evaluate(grid,vinPolicy)
	#e=8.0/28.0
	
	print(evaluate(grid,randomWalk))
	print(evaluate(grid,vinPolicy))#its a bad policy :(
