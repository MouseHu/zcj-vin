import gridworld3 as gw
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
import expBuffer
def randomWalk(status,place):
    return np.random.randint(9)

def vinPolicy(status,place):
    if np.random.random()<e:
        action=np.random.randint(9)
        return action
    S1=torch.Tensor([place[0]]).cuda()
    S2=torch.Tensor([place[1]]).cuda()
    X=status.expand(1, len(status),status[0].shape[0],status[0].shape[1])#.cuda()
    X=X.cuda()
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
def update(expbuffer,vin,oldvin,p=False):
	#(action,state,place,next_state,next_place,reward,over)
	action,X,S1,S2,oldX,oldS1,oldS2,Y=expbuffer.sample()
	#Qmax=torch.Tensor([replay[x[0]][x[1]][4] for x in index]).float() .cuda()


	oldoutputs, _ = oldvin(oldX,oldS1,oldS2,myvin.Config())
	oldouputs=oldoutputs.detach()
	Qmax=(torch.max(oldoutputs,dim=1)[0]).squeeze().cuda()

	outputs, _ = vin(X,S1,S2 ,  myvin.Config())
	print(outputs.shape,action.unsqueeze(1).shape)
	Qvalue=outputs.gather(index=action.unsqueeze(1).long(),dim=1).squeeze().cuda()
	#print(Qvalue.shape)
	#print(Y.shape)

	TDtarget=(Y+gamma*Qmax).cuda()

	criterion = torch.nn.MSELoss(size_average=False)
	loss=criterion(Qvalue,Y).cuda()
	optimizer = optim.RMSprop(VIN.parameters(), lr=myvin.Config().lr, eps=1e-6)
	optimizer.zero_grad()  
	loss.backward()
	optimizer.step()

	if p:
		print(outputs[0],Qvalue[0],TDtarget[0],Y[0].cpu().numpy())
		grid.plot2(X[0].cpu().numpy(),int(S1[0].item()),int(S2[0].item()))	
	return loss
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
	
	VIN=myvin.VIN(myvin.Config())
	VIN=VIN.cuda()
	#VIN.load_state_dict(torch.load("model2/moving-model-9-3920.pkl"))
	print(VIN)
	oldVIN=myvin.VIN(myvin.Config()).cuda()
	oldVIN.load_state_dict(VIN.state_dict())
	grid=gw.GridWorld3_8dir(8,8,nobstacle=4,moving=True)
	e=0
	#print(evaluate(grid,vinPolicy,1000))
	#print(evaluate(grid,randomWalk))
	maxStep=5000000
	episodes=20000
	gamma=0.99
	Tmax=1000
	replay=[]
	max_exp=10000
	learning_begin=1000
	learning_freq=10
	update_freq=1000
	e=0.1
	experience=expBuffer.ExperienceBuffer(myvin.Config(),max_exp)
	print("here")
	#print(evaluate(grid,randomWalk))
	#print(evaluate(grid,vinPolicy,iters=500))
	#time1=ti09, -10me.time()
	#experience
	count=0
	l=0
	s=0
	for k in range(3920,episodes):    
	    #step	
	    #rewards=[]
	    e=100/(k+100)
	
	    state,place,reward,over=grid.reset()
	    #print("begin")
	   # time1=time.time()
	    #if k%10==0:
	    #	grid.plot()
	    time1=time.time()	
	    for i in range(Tmax):
	        count+=1
	        action=vinPolicy(state,place)
	        next_state,next_place,reward,over=grid.step(action)
		experience.add((action,state,place,next_state,next_place,reward,over))
		state=next_state
		place=next_place
		#if i%1000==0:
		 #   print(i)		
		if count<learning_begin:
		    continue
		if count%learning_freq==0 :
		    loss=0
		    #for x in range(3):		
		    loss+=update(experience,VIN,oldVIN)#,True)
		    #if count%1000==0:
			#update(experience,VIN,oldVIN,True)
		    #s+=loss
		    #l+=1
		    #if l%100==0:
			#print("loss",s/100)
			#s=0
			#l=0
		#if count%100==0:
		 #   print("loss",loss/100)
		
		if count%update_freq==0:
			oldVIN.load_state_dict(VIN.state_dict())
			oldVIN.cuda()
		if over:
			#print state
			break
			
	    if k%50==0:
		print("episode",k,time.time()-time1,i,grid.total_reward)
		
	    if count>maxStep:
		break
	#evaluate(grid,randomWalk)
	#evaluate(grid,vinPolicy)
	    
	    if k%200==20:  
		torch.save(VIN.state_dict(),"model/moving-model-9-"+str(k)+".pkl") 
		print("begin eval") 
		print(evaluate(grid,vinPolicy,iters=200))
		
		#print total_reward/iters,time.time()-time2
