##code for Fig 3 of the report



import gym
import matplotlib.pyplot as plt
import random

import numpy as np
random.seed(20)
table = np.random.randn(30,30,2)
table = table.astype(np.float32)
reward_plot = []
reward_mean = []
reward_lr = []
reward_df = [] 
x_plot = []
alpha = [0.1,0.4,0.8,1.0]
df = [0.1,0.4,0.8,1.0]
env = gym.make('CartPole-v1')
##discretisation referred from
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 30)
poleThetaVelSpace = np.linspace(-4, 4, 30)
cartPosSpace = np.linspace(-2.4, 2.4, 30)
cartVelSpace = np.linspace(-4, 4, 30)

#entire function refered from
#https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
##calculate the custom reward
def calculate_reward(state1,next_state1,pav_state1,pav_next_state1):
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

    
for k in range(len(alpha)):
    for j in range(len(df)):
        reward_mean.clear()
        table = np.random.randn(30,30,2)
        table = table.astype(np.float32)
        for i_episode in range(5000):
            x_plot.append(i_episode)
            total_reward =0
            env.reset()
            for t in range(200):
                #env.render()
                if(t==0):
                    action = env.action_space.sample()
                    observation, reward, done, info = env.step(action)
                    ##following 2 lines(62-63) are referred from
                    ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
                    state3 = np.digitize(observation[2],bins=poleThetaSpace)-1
                    state4 = np.digitize(observation[3],bins=poleThetaVelSpace)-1
                    ##storing the actual non-discretised values for calculating reward
                    pa_initial = observation[2]
                    pav_initial = observation[3]

                #get current q value
                q_val = table[state3][state4][action]
                new_observation, reward, done, info = env.step(action)

                ##storing the new non-discretised values for calculating reward
                pa_new = new_observation[2]
                pav_new = new_observation[3]

                ##following 2 lines(78-79) are referred from
                ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
                next_state3 = np.digitize(new_observation[2],bins=poleThetaSpace)-1
                next_state4 = np.digitize(new_observation[3],bins=poleThetaVelSpace)-1
                
                ##find the next best action according to q value
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
                #get the q value for next state
                q_val_new = table[next_state3][next_state4][next_action]
                ##following 2 lines(96-97) are referred from
                ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
                if done and t < 30:
                    reward = -375

                ##send the non-discretised values to calculate custom reward
                reward_use = calculate_reward(pa_initial, pa_new,pav_initial,pav_new)
                #update rule for sarsa
                update = (alpha[k]*(reward_use + df[j]*(q_val_new)-q_val))
                table[state3][state4][action] = table[state3][state4][action] + update

                #pass the future values as current values for next episode
                state3 = next_state3
                state4 = next_state4
                action = next_action
                pa_initial = pa_new
                pav_initial = pav_new
                total_reward = total_reward + reward
                if done:
                    # print("Episode finished after {} timesteps".format(t+1))
                    # print(total_reward)
                    break
            reward_plot.append(t)
            if(i_episode==0):
                reward_mean.append(np.float(t))
            else:
                reward_mean.append(np.float(reward_mean[-1]+t))
        mean_print = []
        for i in range(len(reward_mean)):
            mean_print.append((reward_mean[i])/(i+1))
        reward_mean_copy = mean_print.copy()
        reward_df.append(reward_mean_copy)
        print("DF finished",j)
    print("Alpha finished",k)
            #env.render()
   

env.close()
mean = sum(reward_plot[25::])/((len(reward_plot)-25))
mean_print = []
for i in range(len(reward_mean)):
    mean_print.append((reward_mean[i])/(i+1))

print("Average reward is:",mean)
print(len(reward_plot))

##lines 142-173 are referred from
#https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(reward_df[0],label="0.1")
axs[0,0].plot(reward_df[1],label="0.4")
axs[0,0].plot(reward_df[2],label="0.8")
axs[0,0].plot(reward_df[3],label="1.0")
axs[0, 0].set_title('Alpha = 0.1')
plt.legend()
axs[0,1].plot(reward_df[4],label="0.1")
axs[0,1].plot(reward_df[5],label="0.4")
axs[0,1].plot(reward_df[6],label="0.8")
axs[0,1].plot(reward_df[7],label="1.0")
axs[0, 1].set_title('Alpha = 0.4')
plt.legend()
axs[1,0].plot(reward_df[8],label="0.1")
axs[1,0].plot(reward_df[9],label="0.4")
axs[1,0].plot(reward_df[10],label="0.8")
axs[1,0].plot(reward_df[11],label="1.0")
axs[1, 0].set_title('Alpha = 0.8')
plt.legend()
axs[1,1].plot(reward_df[12],label="0.1")
axs[1,1].plot(reward_df[13],label="0.4")
axs[1,1].plot(reward_df[14],label="0.8")
axs[1,1].plot(reward_df[15],label="1.0")
axs[1, 1].set_title('Alpha = 1.0')
lines = []
labels = []
plt.tight_layout()
axs[0,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[0,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[1,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[1,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
plt.show()

