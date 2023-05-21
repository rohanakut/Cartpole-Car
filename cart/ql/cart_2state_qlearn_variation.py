##code for Fig 5
import gym
import matplotlib.pyplot as plt
import random
import numpy as np

random.seed(20)
#create a Q table of 30 x 30 x 2
table = np.random.randint(0, 5, size=( 30,30, 2))
table = table.astype(np.float32)
reward_plot = []
reward_mean = []
reward_lr = []
reward_df = [] 
x_plot = []
alpha = [0.1,0.4,0.8,1.0]
df = [0.1,0.4,0.8,1.0]
eps = 0.2
env = gym.make('CartPole-v1')

#discretisation referred from
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
poleThetaSpace = np.linspace(-0.20943951, 0.20943951, 30)
poleThetaVelSpace = np.linspace(-4, 4, 30)
cartPosSpace = np.linspace(-2.4, 2.4, 20)
cartVelSpace = np.linspace(-4, 4, 20)

#entire function referred from
##https://towardsdatascience.com/open-ai-gym-classic-control-problems-rl-dqn-reward-functions-16a1bc2b007
def calculate_reward(state1,next_state1):#,pav_state1,pav_next_state1):
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
        table = np.random.randint(0, 5, size=( 50,50, 2))
        table = table.astype(np.float32)
        for i_episode in range(5000):
            observation = env.reset()
            posi_ini = observation[2]
            ##following 2 lines are referred from
            ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
            state1 = np.digitize(observation[2],bins=poleThetaSpace)-1
            state2 = np.digitize(observation[3],bins=poleThetaVelSpace)-1
            done = False
            for t in range(200):
                if(t==0):
                    #first action should be random
                    action = env.action_space.sample()
                else:
                    ## find the best action for the current state greedily
                    action_test = 0
                    opp_test = 1
                    q_val_0 = table[state1][state2][action_test]
                    q_val_1 = table[state1][state2][opp_test]
                    if(q_val_0>q_val_1):
                        action = action_test
                    else:
                        action = opp_test
                q_val = table[state1][state2][action]

                new_observation, reward, done, info = env.step(action)
                #following 2 lines are referred from
                ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
                if done and t < 30:
                    reward = -375
                if done:
                    break
                posi_new = new_observation[2]
                ##following 2 lines are referred from
                ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/sarsa.py
                next_state1 = np.digitize(new_observation[2],bins=poleThetaSpace)-1
                next_state2 = np.digitize(new_observation[3],bins=poleThetaVelSpace)-1
                
                #find the best q value for next state
                action_find = 0
                opp_find = 1
                q_val_0 = table[next_state1][next_state2][action_find]
                q_val_1 = table[next_state1][next_state2][opp_find]
                if(q_val_0>q_val_1):
                    next_action = action_find
                else:
                    next_action = opp_find
                
                q_val_new = table[next_state1][next_state2][next_action]
                #calculate custom reward
                reward_use = calculate_reward(posi_ini,posi_new)
                ##update rule for Q-learning
                update = reward_use + df[j]*(q_val_new)-q_val
                table[state1][state2][action] = q_val + alpha[k]*(update)
                #move to next state
                state1 = next_state1
                state2 = next_state2
                posi_ini = posi_new
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

   

env.close()
mean = sum(reward_plot[25::])/((len(reward_plot)-25))
mean_print = []
for i in range(len(reward_mean)):
    mean_print.append((reward_mean[i])/(i+1))
#print(mean_print)
print("Average reward is:",mean)
print(len(reward_plot))
#all the lines mentioned below are referred from
#https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(reward_df[0],label="0.1")
axs[0,0].plot(reward_df[1],label="0.4")
axs[0,0].plot(reward_df[2],label="0.8")
axs[0,0].plot(reward_df[3],label="1.0")
axs[0, 0].set_title('Learning Rate = 0.1')
plt.legend(loc="lower right")
axs[0,1].plot(reward_df[4],label="0.1")
axs[0,1].plot(reward_df[5],label="0.4")
axs[0,1].plot(reward_df[6],label="0.8")
axs[0,1].plot(reward_df[7],label="1.0")
axs[0, 1].set_title('Learning Rate = 0.4')
plt.legend(loc="lower right")
axs[1,0].plot(reward_df[8],label="0.1")
axs[1,0].plot(reward_df[9],label="0.4")
axs[1,0].plot(reward_df[10],label="0.8")
axs[1,0].plot(reward_df[11],label="1.0")
axs[1, 0].set_title('Learning Rate = 0.8')
plt.legend(loc="lower right")
axs[1,1].plot(reward_df[12],label="0.1")
axs[1,1].plot(reward_df[13],label="0.4")
axs[1,1].plot(reward_df[14],label="0.8")
axs[1,1].plot(reward_df[15],label="1.0")
axs[1, 1].set_title('Learning Rate = 1.0')
plt.legend(loc="lower right")
lines = []
labels = []
plt.tight_layout()
axs[0,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[0,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[1,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
axs[1,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7,loc="lower right")
plt.show()

