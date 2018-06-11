import gridworld as gw
import myvin

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
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
def evaluate(env,policy,iters=500):
	    total_reward=0
	    for i in range(iters):
		status,place,reward,over,action=env.reset()
		while over==False:
		    status,place,reward,over,action=env.step(policy(status,place))
		total_reward+=env.total_reward+0.0

		if i%100==0:
		    print(i)
	    return total_reward/iters
device=1
if len(sys.argv)>1:
   device=int(sys.argv[1])
with torch.cuda.device(device):
	VIN=myvin.VIN(myvin.Config()).cuda()
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
	episodes=2000
	print("here")
	for k in range(epoches):
	    gamma=0.9
	    #step
	    replay=[]
	    #length=[]
	    rewards=[]
	    print("begin")
	    time1=time.time()
	    for i in range(episodes):
		reward=[]
		e=40.0/(40+k)#8.0/(8+k) is not ok why?
		if i==0:
		    experience=grid.run_episode(vinPolicy)#,show=True)
		else:
		    experience=grid.run_episode(vinPolicy)
		#print(experience[:][2])
		#make discounted reward
		reward.append(experience[-1][2])
		for index,state  in enumerate(experience[-2::-1]):
		    reward.append(reward[index]*gamma+state[2])
		replay.append(experience)
		reward=reward[::-1]
		rewards.append(reward)
		if i%100==0:
		    print(i)
	    #update
	    #print(rewards)
	    print("experience",k,time.time()-time1)
	    time1=time.time()	
	    for i in range(20000):
		X=[]
		S1=[]
		S2=[]
		action=[]
		Y=[]#torch.Tensor(reward[::-1])
		for j in range(myvin.Config().batch_size):# sample experience from replay
		    x1=np.random.randint(episodes)
		    x2=np.random.randint(len(replay[x1])-1)
		    #status,place,reward,over,action
		    
		    Y.append(rewards[x1][x2])
		    action.append(replay[x1][x2][4])
		    X.append(replay[x1][x2][0])
		    S1.append(replay[x1][x2][1][0])
		    #print(x1,replay[x1][x2][1][0])
		    S2.append(replay[x1][x2][1][1])
		#print(np.array(X).shape)
		X=torch.from_numpy(np.array(X)).float().cuda()
		S1=torch.from_numpy(np.array(S1)).float().cuda()
		S2=torch.from_numpy(np.array(S2)).float().cuda()
		action=torch.from_numpy(np.array(action)).unsqueeze(dim=1).long().cuda()
		Y=torch.from_numpy(np.array(Y)).float().cuda()#MC target
		#print(action.shape)
		#print(outputs.shape)
		outputs, _ = VIN(X,S1,S2 ,  myvin.Config())
		Qvalue=outputs.gather(index=action,dim=1).squeeze().cuda()
		#print(Qvalue.shape)
		#print(Y.shape)
		criterion = torch.nn.MSELoss(size_average=False)
		loss=criterion(Qvalue,Y).cuda()
		optimizer = optim.RMSprop(VIN.parameters(), lr=2e-2, eps=1e-6)   
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
