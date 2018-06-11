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
    q1,q2=VINinfer(X,S1,S2,myvindyn.Config())
    q1=q1
    q2=q2
    #print(q1)
    #print(q2.shape)
    _,action=torch.max(q2,dim=1)
    action=int(action)    
    #print(action)
    assert 0<=action and action<9
    return action
def vinPredict(status,place,vin):
    if np.random.random()<e:
        action=np.random.randint(8)
        return action
    S1=torch.Tensor([place[0]]).expand(1,1)
    S2=torch.Tensor([place[1]]).expand(1,1)
    X=torch.Tensor(status).expand(1, len(status),status[0].shape[0],status[0].shape[1])
    config=myvindyn.Config()
    q1,q2=vin(X,S1,S2,myvindyn.Config())
    q1=q1
    
    return q1
def update(experience,vin,oldvin,p=False):
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
    for j in range(myvindyn.Config().batch_size):# sample experience from replay
        x=np.random.randint(len(experience))
        #status,place,reward,over,action
        while experience[x][6]==True:
            x=np.random.randint(len(experience))
        Y.append(experience[x][5])
        action.append(experience[x][0])
        X.append(experience[x][1])
        oldX.append(experience[x][3])
        S1.append(experience[x][2][0])
        S2.append(experience[x][2][1])
        oldS1.append(experience[x][4][0])
        oldS2.append(experience[x][4][1])
        #index.append((x1,x2+1))

    X=torch.from_numpy(np.array(X)).float()#do not change it to torch.Tensor(X).float()
    S1=torch.from_numpy(np.array(S1)).float()#.unsqueeze(1)
    S2=torch.from_numpy(np.array(S2)).float()#.unsqueeze(1)

    oldX=torch.from_numpy(np.array(oldX)).float()
    oldS1=torch.from_numpy(np.array(oldS1)).float()#.unsqueeze(1)
    oldS2=torch.from_numpy(np.array(oldS2)).float()#.unsqueeze(1)
    #print("here",S1.shape)
    action=torch.from_numpy(np.array(action)).unsqueeze(dim=1).long()

    Y=torch.from_numpy(np.array(Y)).float()
    #Qmax=torch.Tensor([replay[x[0]][x[1]][4] for x in index]).float() 


    oldoutputs, _ = oldvin(oldX,oldS1,oldS2 ,  myvindyn.Config())
    oldouputs=oldoutputs.detach()
    Qmax=(torch.max(oldoutputs,dim=1)[0]).squeeze()

    outputs, _ = vin(X,S1,S2 ,  myvindyn.Config())
    Qvalue=outputs.gather(index=action,dim=1).squeeze()
    #print(Qvalue.shape)
    #print(Y.shape)

    TDtarget=(Y+gamma*Qmax)
    bellman_error=-(TDtarget-Qvalue)
    
    optimizer = optim.RMSprop(VINinfer.parameters(), lr=myvindyn.Config().lr, eps=1e-6) 
    optimizer.zero_grad()  
    Qvalue.backward(bellman_error.data)
    optimizer.step()

    if p:
        print(outputs[0],Qvalue[0],TDtarget[0],Y[0].cpu().numpy())
        grid.plot2(X[0].cpu().numpy(),int(S1[0].item()),int(S2[0].item()))
    return loss
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
device=0


VINinfer=myvindyn.VINinfer(myvindyn.Config())
#VINinfer.load_state_dict(torch.load("vin_8x8.pth",map_location='cpu'))
print(VINinfer)
oldVINinfer=myvindyn.VINinfer(myvindyn.Config())
oldVINinfer.load_state_dict(VINinfer.state_dict())
grid=gw.GridWorld2_8dir(8,8,nobstacle=4,moving=False)

#print(evaluate(grid,vinPolicy,1000))
#print(evaluate(grid,randomWalk))
maxStep=5000000
episodes=20000
gamma=0.99
Tmax=100
replay=[]
max_exp=100000
learning_begin=10000
learning_freq=4
update_freq=4000
e=0
experience=[]
print("here")
#print(evaluate(grid,randomWalk))
#print(evaluate(grid,vinPolicy,iters=500))
#time1=ti09, -10me.time()
#experience
count=0
l=0
s=0
#print(evaluate(grid,vinPolicy,iters=500))
print(evaluate(grid,randomWalk,iters=500))
for k in range(episodes):    
    #step	
    #rewards=[]
    e=0.1*(1-(k+0.0)/episodes)

    state,place,reward,over=grid.reset()
    #print("begin")
  # time1=time.time()

#if k%10==0:
    #grid.plot()
    time1=time.time()
    tmp_experience=[]
    for i in range(Tmax):
        
        #print(len(experience),count)
        if len(experience) < learning_begin:
            action=randomWalk(state,place)
        else:
            action=vinPolicy(state,place)
        next_state,next_place,reward,over=grid.step(action)
        tmp_experience.append((action,state,place,next_state,next_place,reward,over))
        
        state=next_state
        place=next_place
    
    #if i%1000==0:
    #   print(i)		
        if len(experience)<learning_begin:
            if over:
                break
            continue
        #print(len(experience),"begin")
        count+=1
        if count%learning_freq==0 :
            loss=0
            for x in range(10):		
                loss+=update(experience,VINinfer,oldVINinfer)#,True)
            #if count%20000==0:
             #   print(loss)
            #if count%1000==0:
            #update(experience,VINinfer,oldVINinfer,True)
            #s+=loss
            #l+=1
            #if l%100==0:
            #print("loss",s/100)
            #s=0
            #l=0
        #if count%100==0:
     #   print("loss",loss/100)

        if count%update_freq==0:
            oldVINinfer.load_state_dict(VINinfer.state_dict())
            print("update")
            #oldVINinfer
        if over:
            #print state
            break
    if tmp_experience[-1][6]==True:
        #print(len(tmp_experience))
        experience+=tmp_experience
        while len(experience)>max_exp:
            experience.pop(0)  
    if k%10==0:
        print("episode",k,time.time()-time1,i,grid.total_reward)

    if count>maxStep:
        break
#evaluate(grid,randomWalk)
#evaluate(grid,vinPolicy)

    if k%500==0:  
        torch.save(VINinfer.state_dict(),"model/moving-model-vindyn-"+str(k)+".pkl") 
        print("begin eval") 
        if k is not 0:
            print(evaluate(grid,vinPolicy,iters=200))
        else:
            print(evaluate(grid,vinPolicy,iters=200))

    #print total_reward/iters,time.time()-time2