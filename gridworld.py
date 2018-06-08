import numpy as np
import matplotlib.pyplot as plt
import math
class Obstacle(object):
    def __init__(self,width,height,nobstacle,avoid,moving=True):
        self.width=width
        self.height=height
        self.num=0# number of obstacles
        self.obstacles=[]
        self.dir_list=[[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0],[-1,-1],[0,-1],[1,-1]]
        for i in range(nobstacle):
            self.genObstacle(avoid)
        self.moving=moving

    def checkCollision(self,pos,avoid):#check Collision
        for i in avoid:
                if i==pos:
                    return True
        return False
    def genObstacle(self,avoid):
        if self.num>=self.width*self.height-len(avoid):
            print("error:no place to hold obstacles")
            return
        flag=True
        while flag:
            s1,s2=(np.random.randint(self.width),np.random.randint(self.height))
            flag=self.checkCollision((s1,s2),self.obstacles+avoid)
        self.obstacles.append((s1,s2))
        self.num+=1
        return
    def move(self,avoid):
        if self.moving==False:
            return
        for index,ob in enumerate(self.obstacles):
            #print(self.dir_list,np.random.randint(9))
            move=self.dir_list[np.random.randint(9)]
            new_pos=(max(min(ob[0]+move[0],self.width-1),0),max(min(ob[1]+move[1],self.height-1),0))
            if self.checkCollision(new_pos,avoid)==False:
                self.obstacles[index]=new_pos
        return
class GridWorld_8dir(object):#8dir
    def __init__(self,width=10,height=10,nobstacle=3,moving=True):
        self.dir_list=[[-1,1],[0,1],[1,1],[-1,0],[0,0],[1,0],[-1,-1],[0,-1],[1,-1]]#0 up 1 down 2 left 3 right
        
        self.width=width
        self.height=height
        self.moving=moving
        self.nobstacle=nobstacle
        
        self.step_reward=-1;
        self.goal_reward=100;
    
        self.reset()    
    def reset(self):
        while True:
            self.init_place=(np.random.randint(self.width),np.random.randint(self.height))
            self.goal_place=(np.random.randint(self.width),np.random.randint(self.height))
            #while self.goal_place==self.init_place:
            #    self.goal_place=(np.random.randint(width),np.random.randint(height))
            if (abs(self.init_place[0]-self.goal_place[0])+abs(self.init_place[1]-self.goal_place[1]))>math.sqrt(self.width**2+self.height**2):
                break
        #print(abs(self.init_place[0]-self.goal_place[0])+abs(self.init_place[1]-self.goal_place[1]))
        self.obstacles=Obstacle(self.width,self.height,self.nobstacle,[self.init_place,self.goal_place],self.moving)
        self.place=self.init_place
        self.total_reward=0;
        self.over=False
        
        return self.status(),self.place,self.total_reward,self.over
        
    def status(self):#inbulid
        reward_map=np.full((self.width,self.height),0)
        obstacle_map=np.full((self.width,self.height),0)
        #array[self.place]=1
        #print(self.init_place, array[self.init_place])
        reward_map[self.goal_place]=self.goal_reward#prior knewledge
        for ob in self.obstacles.obstacles:
            #print(ob)
            obstacle_map[ob]=1
        #print(array)
        return [reward_map,obstacle_map]
    
    def show(self):
        array=np.full((self.width,self.height),0)
        array[self.place]=1
        #print(self.init_place, array[self.init_place])
        array[self.goal_place]=2
        for ob in self.obstacles.obstacles:
            #print(ob)
            array[ob]=-1
        #print(array)
        return array
    
    def step(self,direction):
        current_reward=self.total_reward
        if self.over:
            return self.status(),self.goal_place,0,True
        #if meet obstacle, stop at origin place
        
        self.obstacles.move([self.place,self.goal_place])
        
        assert (0<=direction and direction<9)
        new_place=(max(min(self.place[0]+self.dir_list[direction][0],self.width-1),0),max(min(self.place[1]+self.dir_list[direction][1],self.height-1),0))
        if self.obstacles.checkCollision(new_place,self.obstacles.obstacles)==False:
            #print("here!")
            self.place=new_place
            
        self.total_reward+=self.step_reward
        
        if self.place==self.goal_place:
            self.over=True
            self.total_reward+=self.goal_reward
        return self.status(),self.place,self.total_reward-current_reward,self.over
    
    def run_episode(self,policy,show=False,Tmax=2000):
        status,place,reward,over=self.reset()
        experience=[]
        count=0
        while over==False:
            count+=1
	    action=policy(status,place)
	    experience.append((status,place,reward,over,action))
            status,place,reward,over=self.step(action)
            
            if show==True and count%100==0:
                self.plot()
	    if count>=Tmax:
		break
	experience.append((status,place,reward,over,-1))
        #print(place,reward,over)
        #self.clean()
        return experience
    def run_episode2(self,policy,show=False):
        status,place,reward,over=self.reset()
        experience=[]
        count=0
        while over==False:
            count+=1
	    print(policy(status,place))
	    action,q=policy(status,place)
            status,place,reward,over=self.step(action)
	    Q=int(torch.max(q,dim=1)[1])
            experience.append((status,place,reward,over,action,Q))
            if show==True and count%100==0:
                self.plot()
        #print(place,reward,over)
        #self.clean()
        return experience
    def sample(self):
        return np.random.randint(9)
    
    def plot(self):
        plt.figure(figsize=(5, 5))
        ax=plt.gca()  
        ax.set_xticks(np.linspace(0,self.width,self.width+1))  
        ax.set_yticks(np.linspace(0,self.height,self.height+1))  
        
        plt.grid(True)  
        plt.xlim((0, self.width))
        plt.ylim((0, self.height))
        matrix=self.show()
        x=np.arange(0, self.width, 0.01)
        for j in range(self.height):
            y1=np.array([j]*len(x))
            y2=np.array([j+1]*len(x))
            for i in range(self.width):
                if matrix[i,j]==0:
                    continue
                if matrix[i,j]==-1:
                    plt.fill_between(x,y1,y2,where=(i<=x) & (x<=i+1),facecolor='black')
                elif matrix[i,j]==1:
                    plt.fill_between(x,y1,y2,where=(i<=x) & (x<=i+1),facecolor='blue')
                else:
                    plt.fill_between(x,y1,y2,where=(i<=x) & (x<i+1),facecolor='red')
        plt.show()
        return
