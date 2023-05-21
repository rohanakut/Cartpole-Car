import gym
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(20)
## creates a state space fo 30 x 30 x 2
table = np.random.randn(30,30,2)
table = table.astype(np.float32)
reward_plot = []
reward_mean = []
x_plot = []
env = gym.make('CartPole-v1')

#discretisation refered from
## https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py 
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 30)
poleThetaVelSpace = np.linspace(-4, 4, 30)
cartPosSpace = np.linspace(-2.4, 2.4, 30)
cartVelSpace = np.linspace(-4, 4, 30)

#entire function refered from
#https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
##calculate the custom reward
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
    

for i_episode in range(3000):
    x_plot.append(i_episode)
    total_reward =0
    env.reset()
    for t in range(200):
        #env.render()
        if(t==0):
            #start with a random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            ##Following 2 lines are referred from
            ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
            state3 = np.digitize(observation[2],bins=poleThetaSpace)-1
            state4 = np.digitize(observation[3],bins=poleThetaVelSpace)-1
            #save the non-discretised pole angle value to calculate reward
            pa_initial = observation[2]
            pav_initial = observation[3]
       
        ##save the current state q value
        q_val = table[state3][state4][action]

        new_observation, reward, done, info = env.step(action)
        #save the next state non-discretised pole angle value to calculate reward
        pa_new = new_observation[2]
        pav_new = new_observation[3]
        
        ##following 2 lines are refered from
        ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
        next_state3 = np.digitize(new_observation[2],bins=poleThetaSpace)-1
        next_state4 = np.digitize(new_observation[3],bins=poleThetaVelSpace)-1
        
        ##find the best next action
        if(action==0):
            opp = 1
        else:
            opp=0
        q_val_opp_test = table[next_state3][next_state4][opp]
        q_val_test = table[next_state3][next_state4][action]
        if(q_val_opp_test>=q_val_test):
            next_action = opp
        else:
            next_action = action
        
        #find the q-value for next state
        q_val_new = table[next_state3][next_state4][next_action]
        
        #following 2 lines are referred from
        ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
        if done and t < 30:
            reward = -375
        
        #send the non-discretised pole angle values to calculate custom reward
        reward_use = calculate_reward(pa_initial, pa_new)
        
        #update rule for sarsa.
        ## Note: The learning rate and discount factor have been calculated as mentioned in the report
        update = (0.8*(reward_use + 0.8*(q_val_new)-q_val))
        table[state3][state4][action] = table[state3][state4][action] + update

        #move to the next state and next action.
        state3 = next_state3
        state4 = next_state4
        action = next_action
        pa_initial = pa_new
        pav_initial = pav_new
        total_reward = total_reward + reward
        
        if done:
            break

    reward_plot.append(t)
     ## finding the mean reward per episode(orange line in the plots)
    if(i_episode==0):
        reward_mean.append(np.float(t))
    else:
        reward_mean.append(np.float(reward_mean[-1]+t))
    #env.render()
   

env.close()
mean = sum(reward_plot[25::])/((len(reward_plot)-25))
mean_print = []
for i in range(len(reward_mean)):
    mean_print.append((reward_mean[i])/(i+1))
print("Average reward is:",mean)
print(len(reward_plot))

##following 5 lines are referred from 
#https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
plt.plot(reward_plot,label = "Episode Reward")
plt.plot(mean_print,label = "Average Reward")
plt.legend(loc="best")
plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)

#https://stackoverflow.com/questions/52666450/put-text-label-at-the-end-of-every-line-plotted-through-matplotlib-with-three-di
plt.annotate(xy=(len(mean_print),mean_print[-1]), xytext=(5,0), textcoords='offset points', weight='bold',fontsize=14,text="Average reward:\n %s"%(round(mean, 2)), va='center')

#following 3 lines
#https://stackoverflow.com/questions/46599171/dashed-lines-from-points-to-axes-in-matplotlib
plt.vlines(len(mean_print), 0, mean_print[-1], linestyle="dashed",colors="red")
plt.hlines(mean_print[-1], 0, len(mean_print), linestyle="dashed",colors="red")
plt.show()