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
def vinPredict(status,place,vin):
    if np.random.random()<e:
        action=np.random.randint(9)
        return action
    S1=torch.Tensor([place[0]]).cuda()
    S2=torch.Tensor([place[1]]).cuda()
    X=torch.Tensor(status).expand(1, len(status),status[0].shape[0],status[0].shape[1]).cuda()
    config=myvin.Config()
    q1,q2=vin(X,S1,S2,myvin.Config())
    q1=q1.cuda()
    
    return q1
def update(experience,vin,oldvin):
	#(action,state,place,next_state,next_place,reward,over)
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
	    x=np.random.randint(len(experience))
	    #status,place,reward,over,action
 
	    Y.append(experience[x][5])
	    action.append(experience[x][0])
	    X.append(experience[x][1])
	    oldX.append(experience[x][3])
	    S1.append(experience[x][2][0])
	    S2.append(experience[x][2][1])
	    oldS1.append(experience[x][4][0])
	    oldS2.append(experience[x][4][1])
	    #index.append((x1,x2+1))

	X=torch.from_numpy(np.array(X)).float().cuda()#do not change it to torch.Tensor(X).float()
	S1=torch.from_numpy(np.array(S1)).float().cuda()
	S2=torch.from_numpy(np.array(S2)).float().cuda()
	oldS1=torch.from_numpy(np.array(oldS1)).float().cuda()
	oldS2=torch.from_numpy(np.array(oldS2)).float().cuda()
	oldX=torch.from_numpy(np.array(oldX)).float().cuda()
	action=torch.from_numpy(np.array(action)).unsqueeze(dim=1).long().cuda()
	Y=torch.from_numpy(np.array(Y)).float().cuda()
	#Qmax=torch.Tensor([replay[x[0]][x[1]][4] for x in index]).float() .cuda()


	oldoutputs, _ = oldvin(oldX,oldS1,oldS2 ,  myvin.Config())
	Qmax=(torch.max(oldoutputs,dim=1)[0]).squeeze().cuda()

	outputs, _ = vin(X,S1,S2 ,  myvin.Config())
	Qvalue=outputs.gather(index=action,dim=1).squeeze().cuda()
	#print(Qvalue.shape)
	#print(Y.shape)
	TDtarget=(Y+gamma*Qmax).cuda()
	criterion = torch.nn.MSELoss(size_average=False)
	loss=criterion(Qvalue,Y).cuda()
	optimizer = optim.RMSprop(VIN.parameters(), lr=myvin.Config().lr, eps=1e-6) 
	optimizer.zero_grad()  
	loss.backward()
	# Update params
	optimizer.step()
def evaluate(env,policy,iters=5000):
	total_reward=0
	time2=time.time()
	for i in range(iters):
		status,place,reward,over=env.reset()
		t=0
		while over==False and t<2000:
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
	print(VIN)
	oldVIN=copy.deepcopy(VIN).cuda()
	grid=gw.GridWorld_8dir(8,8,nobstacle=4,moving=False)

	maxStep=1000000
	episodes=10000
	gamma=0.99
	Tmax=5000
	replay=[]
	max_exp=50000
	learning_begin=20000
	learning_freq=4
	update_freq=5000
	e=0.1
	experience=[]
	print("here")
	#print(evaluate(grid,randomWalk))
	#print(evaluate(grid,vinPolicy,iters=500))
	#time1=ti09, -10me.time()
	#experience
	count=0
	for k in range(episodes):    
	    #step	
	    #rewards=[]
	    e=50.0/(k+50)

	    state,place,reward,over=grid.reset()
	    #print("begin")
	   # time1=time.time()
	    
	    time1=time.time()	
	    for i in range(Tmax):
	        count+=1
	        action=vinPolicy(state,place)
	        next_state,next_place,reward,over=grid.step(action)
		experience.append((action,state,place,next_state,next_place,reward,over))
		if len(experience)>max_exp:
			experience.pop(0)
		state=next_state
		place=next_place
		#if i%1000==0:
		 #   print(i)		
		if count<learning_begin:
		    continue
		if count%learning_freq==0 :		
		    update(experience,VIN,oldVIN)
		    #print("update")
		
		if count%update_freq==0:
			oldVIN.load_state_dict(VIN.state_dict())
			oldVIN.cuda()
		if over:
			#print state
			break
			
	    if k%10==0:
		print("episode",k,time.time()-time1,i,grid.total_reward)
	    if count>maxStep:
		break
	#evaluate(grid,randomWalk)
	#evaluate(grid,vinPolicy)
	    
	    if k%1000==20:   
		print("begin eval") 
		iters=10
		#print(evaluate(grid,vinPolicy,iters=100))#its a bad policy :(
		total_reward=0
		time2=time.time()
		for x in range(iters):
			state,place,reward,over=grid.reset()
			#t=0
			e=0
			for y in range(Tmax):
				action=vinPolicy(state,place)
				next_state,next_place,reward,over=grid.step(action)
				state=next_state
				place=next_place
				#t+=1
				if over:
					break
			total_reward+=grid.total_reward+0.0
			print(grid.total_reward)
			if x%5==0:
				print(x)
		
		print total_reward/iters,time.time()-time2
	
	
