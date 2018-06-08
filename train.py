import gridworld as gw
import myvin

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

#import torchvision.transforms as transforms
grid=gw.GridWorld_8dir(nobstacle=3)
#grid.show()
#grid.plot()
#for _ in range(100):
#    grid.step(grid.sample())
#grid.plot()
def randomWalk(status,place):
    return np.random.randint(9)

#evaluate(grid,randomWalk)

VIN=myvin.VIN(myvin.Config())
def vinPolicy(status,place):
    if np.random.random()<e:
	action=np.random.randint(9)
	return action
    S1=torch.Tensor([place[0]])
    S2=torch.Tensor([place[1]])
    X=torch.Tensor(status).expand(1, len(status),status[0].shape[0],status[0].shape[1])
    config=myvin.Config()
    q1,q2=VIN(X,S1,S2,myvin.Config())
    #print(q1)
    #print(q2.shape)
    _,action=torch.max(q1,dim=1)
    action=int(action)    
    #print(action)
    assert 0<=action and action<9
    return action
epoches=100
episodes=200
print("here")
for k in range(epoches):
    gamma=0.9
    #step
    replay=[]
    #length=[]
    rewards=[]
    print("begin")
    for i in range(episodes):
	reward=[]
	e=10.0/(10+k)#8.0/(10+k) is not ok why?
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
	if i%20==0:
	    print(i)
    #update
    #print(rewards)
    print("experience",k)
    for i in range(2000):
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
	X=torch.from_numpy(np.array(X)).float()
	S1=torch.from_numpy(np.array(S1)).float()
	S2=torch.from_numpy(np.array(S2)).float()
	action=torch.from_numpy(np.array(action)).unsqueeze(dim=1).long()
	Y=torch.from_numpy(np.array(Y)).float()
	#print(action.shape)
	#print(outputs.shape)
	outputs, _ = VIN(X,S1,S2 ,  myvin.Config())
	Qvalue=outputs.gather(index=action,dim=1).squeeze()
	#print(Qvalue.shape)
	#print(Y.shape)
	criterion = torch.nn.MSELoss(size_average=False)
	loss=criterion(Qvalue,Y)
	optimizer = optim.RMSprop(VIN.parameters(), lr=1e-2, eps=1e-6)   
	loss.backward()
	# Update params
	optimizer.step()
	if i%50==0:
	    print(i)
    print("update",k)
#evaluate(grid,randomWalk)
#evaluate(grid,vinPolicy)
#e=8.0/28.0
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
print(evaluate(grid,randomWalk))
print(evaluate(grid,vinPolicy))#its a bad policy :(
