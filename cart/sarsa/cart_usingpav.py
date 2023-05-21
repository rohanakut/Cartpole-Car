## code for single state SARSA

import gym
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(20)
#create a state space of 30 x 2
table = np.random.randint(0, 5, size=( 30, 2))
table = table.astype(np.float32)
reward_plot = []
reward_mean = []
x_plot = []
env = gym.make('CartPole-v0')

## discretisation refered from 
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 30)
poleThetaVelSpace = np.linspace(-4, 4, 10)
cartPosSpace = np.linspace(-2.4, 2.4, 10)
cartVelSpace = np.linspace(-4, 4, 10)

#entire function referred from
#https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
def calculate_reward(state1,next_state1):
    reward_send = 0
    if(state1>0):
        if(next_state1>state1):
            reward_send = -1
        if(next_state1<state1):
            reward_send = 1
    if(state1<0):
        if(next_state1>state1):
            reward_send = 1
        if(next_state1<state1):
            reward_send = -1
    #print("Reward to be given:",reward_send)
    return reward_send


for i_episode in range(20000):
    x_plot.append(i_episode)
    total_reward =0
    observation = env.reset()
    for t in range(100):
        #env.render()
        epsilon = random.randint(1, 100)
       # print(observation)
        if(t==0):
            #first action will be random
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            pav_initial = observation[2]
            
            #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
            state2 = np.digitize(observation[2],bins=poleThetaSpace)-1
            #save current state q value
            q_val = table[state2][action]
            total_reward = total_reward + reward
        
        if done:
            break
        
        #save non-discretised pole angle for calculating reward
        pav = observation[2]

        # find the best action
        if(action==0):
            opp = 1
        else:
            opp=0
        q_val_opp = table[state2][opp]
        q_val = table[state2][action]
        if(q_val_opp>=q_val):
            next_action = opp
        else:
            next_action = action
        
        observation, reward, done, info = env.step(next_action)
        #find the non discretised pole angle for next state
        pav_new = observation[2]

        #following 2 lines are referred from
        ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
        if done and t < 30:
            reward = -375
        
        #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
        next_state2 = np.digitize(observation[2],bins=np.linspace(-0.20943951, 0.20943951, 30))-1
        
        reward_use = calculate_reward(pav_initial, pav_new)

        #update rule for SARSA
        update = (0.2*(reward_use + 0.7*(table[next_state2][next_action])-table[state2][action]))
        table[state2][action] = table[state2][action] + update

        #choosing the next state and action
        state2 = next_state2
        action = next_action
        pav_initial = pav_new
        total_reward = total_reward + reward

    reward_plot.append(t)
    if(i_episode==0):
        reward_mean.append(np.float(t))
    else:
        reward_mean.append(np.float(reward_mean[-1]+t))

env.close()
mean = sum(reward_plot[25::])/((len(reward_plot)-25))
mean_print = []
for i in range(len(reward_mean)):
    mean_print.append((reward_mean[i])/(i+1))
print(mean_print)
print("Average reward is:",mean)
print(len(reward_plot))
#following 5 lines refrred from
#https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
plt.plot(reward_plot,label = "Episode Reward")
plt.plot(mean_print,label = "Average Reward")
plt.legend(loc="best")
plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)

#https://stackoverflow.com/questions/52666450/put-text-label-at-the-end-of-every-line-plotted-through-matplotlib-with-three-di
plt.annotate(xy=(len(mean_print),mean_print[-1]), xytext=(5,0), textcoords='offset points', weight='bold',fontsize=14,text="Average reward:\n %s"%(round(mean, 2)), va='center')
plt.show()