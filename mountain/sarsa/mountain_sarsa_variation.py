
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')

#increased the length of episodes. Refered from
#https://www.reddit.com/r/reinforcementlearning/comments/agd6j4/how_to_change_episode_length_and_reward/
env._max_episode_steps = 1000

##discretisation refered from
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
position = np.linspace(-1.2, 0.5, 20)
velocity = np.linspace(-0.07, 0.07, 20)

#q table size of 20 x 20 x3 declared
table = np.random.randint(0, 5, size=( 20,20, 3))
table = table.astype(np.float32)
reward_plot = []
accumulate = []
reward_mean = []
reward_lr = []
reward_df = [] 
x_plot = []
alpha = [0.1,0.4,0.8,1.0]
df = [0.1,0.4,0.8,1.0]
goal = 0
eps = 1.0
#EPSILON_DECREMENTER = EPSILON/(1000//10)



def choose_action(state1, state2):
    q_val1 = np.int(table[state1][state2][0])
    q_val2 = np.int(table[state1][state2][1])
    q_val3 = np.int(table[state1][state2][2])
    #following line referred from
    ##https://github.com/ikvibhav/reinforcement_learning/blob/master/code/mountain_car/mountain_car_sarsa.py
    action = (np.argmax([q_val1,q_val2,q_val3]))
    return action

#custom reward refered from
#https://shiva-verma.medium.com/solving-reinforcement-learning-classic-control-problems-openaigym-1b50413265dd
def get_reward(state):
    reward = 0
    if state >= 0.5:
       # print("Car has reached the goal")
        return 100
    if state > -0.4 and state<0.3:
        return (1+state)
    if state>=0.3:
        return (1+state)**2
    if state<-0.3 and state>-0.7:
        return -1
    return reward

for k in range(len(alpha)):
    for j in range(len(df)):
        reward_mean.clear()
        eps = 1.0
        goal = 0
        flag = 0
        table = np.random.randint(0, 5, size=( 20,20, 3))
        table = table.astype(np.float32)
        for i_episode in range(5000):
            total_reward = 0
            observation = env.reset()
            posi_curr = observation[0]
            vel_curr = observation[1]
            ##following 2 lines referred from
            ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
            state1 = np.digitize(posi_curr,bins=position)-1
            state2 = np.digitize(vel_curr,bins=velocity)-1

            flag = 0

            #epsilon greedy policy
            #referred from 
            ###https://github.com/ikvibhav/reinforcement_learning/blob/master/code/mountain_car/mountain_car_sarsa.py
            if np.random.random() > 0.2:
                action = choose_action(state1,state2)
            else:
                action = np.random.randint(0, env.action_space.n)
            
            for t in range(1000):
                new_observation, reward, done, info = env.step(action)
                vel_new = new_observation[1]
                posi_new = new_observation[0]

                ##following 2 lines referred from
                ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
                next_state1 = np.digitize(new_observation[0],bins=position)-1
                next_state2 = np.digitize(new_observation[1],bins=velocity)-1

                #epsilon greedy policy
                #referred from 
                ###https://github.com/ikvibhav/reinforcement_learning/blob/master/code/mountain_car/mountain_car_sarsa.py
                if np.random.random() > eps:
                    next_action = choose_action(next_state1,next_state2)
                else:
                    next_action = np.random.randint(0, env.action_space.n)
                    
                ##get q values
                q_val = table[state1][state2][action]
                q_val_new = table[next_state1][next_state2][next_action]

                #generate custom reward
                reward_use = get_reward(posi_new)

                #if target position reached then chcek for termination condition
                if(reward_use==100):
                    goal = goal+1
                    print("Goal is:",goal)
                    print("Episode is:",i_episode)
                    done = True
                    table[state1][state2][action] = reward_use
                    #terminate if car reached goal 5 times
                    if(goal >= 5):
                        flag = 1
                        print(total_reward)
                    total_reward = total_reward + reward_use
                    break

                #update rule for SARSA
                update = alpha[k]*(reward_use + df[j]*(q_val_new)-q_val)
                table[state1][state2][action] = table[state1][state2][action] + update

                ##move to next state
                state1 = next_state1
                state2 = next_state2
                action = next_action

                vel_curr = vel_new
                posi_curr = posi_new
                total_reward = total_reward + reward_use
            if(total_reward==0):
                accumulate.append(reward_use)
            
            #if car is getting stuck in a state then increase randomness
            if(len(accumulate)>10):
                eps = 0.8
                accumulate.clear()

            #epsilon decay strategy referred from
            #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
            eps = eps - 2/5000 if eps > 0.01 else 0.01

            reward_plot.append(total_reward)
            #calculate mean reward per episode
            if(i_episode==0):
                reward_mean.append(np.float(total_reward))
            else:
                reward_mean.append(np.float(reward_mean[-1]+total_reward))

           
            #terminate training if car reached goal 5 times
            if(flag==1):
                break
            mean_print = []

            #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py  
            if(i_episode%500==0):
                print('episode ', i_episode, 'score ', total_reward, 'epsilon %.3f' % eps)

            #normalisation of reward to calculate mean reward
            for i in range(len(reward_mean)):
                mean_print.append((reward_mean[i])/(i+1))

            reward_mean_copy = mean_print.copy()
        reward_df.append(reward_mean_copy)
        print(len(reward_mean_copy))
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

#all the following lines referred from
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
axs[0,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7)
axs[0,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7)
axs[1,0].legend(title="Discount Factor",fontsize=7,title_fontsize=7)
axs[1,1].legend(title="Discount Factor",fontsize=7,title_fontsize=7)
plt.show()

