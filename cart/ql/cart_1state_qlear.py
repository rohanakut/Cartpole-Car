import gym
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(20)
#creates a table of dimensions 20 x 2
table = np.random.randint(0, 5, size=( 20, 2))
table = table.astype(np.float32)
reward_plot = []
reward_mean = []
x_plot = []
eps = 0.3
env = gym.make('CartPole-v1')

#discretisation referred from 
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 20)
poleThetaVelSpace = np.linspace(-4, 4, 20)
cartPosSpace = np.linspace(-2.4, 2.4, 2)
cartVelSpace = np.linspace(-4, 4, 20)

#calculate custom reward
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
    return reward_send

for i_episode in range(20000):
    observation = env.reset()
    posi_ini = observation[2]
    ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
    state1 = np.digitize(posi_ini,bins=poleThetaSpace)-1
    done = False
    for t in range(200):

        if(t==0):
            action = env.action_space.sample()
        else:
            #select action greedily according to the best q-value
            action_test = 0
            opp_test = 1
            q_val_0 = table[state1][action_test]
            q_val_1 = table[state1][opp_test]
            if(q_val_0>q_val_1):
                action = action_test
            else:
                action = opp_test
        if done:
            break
       
        q_val = table[state1][action]
        new_observation, reward, done, info = env.step(action)

        #following 2 lines are referred from
        ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
        if done and t < 30:
            reward = -375
        
        posi_new = new_observation[2]
        ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
        next_state1 = np.digitize(new_observation[2],bins=poleThetaSpace)-1

        ##find the best action for next state. this is done to find the action with highest q value
        action_find = 0
        opp_find = 1
        q_val_0 = table[next_state1][action_find]
        q_val_1 = table[next_state1][opp_find]
        if(q_val_0>q_val_1):
            next_action = action_find
        else:
            next_action = opp_find
            #print(action)
        
        #select qvalue that has the highest value for net state
        q_val_new = table[next_state1][next_action]

        #find custom reward
        reward_use = calculate_reward(posi_ini,posi_new)

        #update rule for Q-learning
        update = reward + 0.7*(q_val_new)-q_val
        table[state1][action] = q_val + 0.2*(update)

        #move to next state
        state1 = next_state1
        posi_ini = posi_new

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
#following 5 lines
#https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
plt.plot(reward_plot,label = "Episode Reward")
plt.plot(mean_print,label = "Average Reward")
plt.legend(loc="best")
plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)
#https://stackoverflow.com/questions/52666450/put-text-label-at-the-end-of-every-line-plotted-through-matplotlib-with-three-di
plt.annotate(xy=(len(mean_print),mean_print[-1]), xytext=(5,0), textcoords='offset points', weight='bold',fontsize=14,text="Average reward:\n %s"%(round(mean, 2)), va='center')
plt.show()

