
import gym
import math
import numpy as np
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')
#increase length of episode. Referred from
#https://www.reddit.com/r/reinforcementlearning/comments/agd6j4/how_to_change_episode_length_and_reward/
env._max_episode_steps = 1000

#discretisation referred from 
##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
position = np.linspace(-1.2, 0.5, 20)
velocity = np.linspace(-0.07, 0.07, 20)
#creates a q table of size 20 x 20 x 3
table = np.random.randint(0, 5, size=( 20,20, 3))
table = table.astype(np.float32)
reward_plot = []
accumulate = []
reward_mean = []
goal = 0
eps = 1.0



def choose_action(state1, state2):
    q_val1 = np.int(table[state1][state2][0])
    q_val2 = np.int(table[state1][state2][1])
    q_val3 = np.int(table[state1][state2][2])
    #following line referred from
    ##https://github.com/ikvibhav/reinforcement_learning/blob/master/code/mountain_car/mountain_car_sarsa.py
    action = (np.argmax([q_val1,q_val2,q_val3]))
    return action

#entire function referred from
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

for i_episode in range(10000):
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
        #following 2 lines referred from 
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

        #get q values for current and next states
        q_val = table[state1][state2][action]
        q_val_new = table[next_state1][next_state2][next_action]

        #get custom reward
        reward_use = get_reward(posi_new)

        #check if car reached the target position
        if(reward_use==100):
            #if(i_episode>5000):
            goal = goal+1
            print("Goal is:",goal)
            print("Episode is:",i_episode)
            done = True
            table[state1][state2][action] = reward_use
            #terminate if car reached goal 10 times
            if(goal >= 10):
                flag = 1
                total_reward = total_reward + reward_use
                print(total_reward)
            break

        #update rule for SARSA
        update = 0.8*(reward_use + 0.95*(q_val_new)-q_val)
        table[state1][state2][action] = table[state1][state2][action] + update

        #move to next state and action
        state1 = next_state1
        state2 = next_state2
        action = next_action
        vel_curr = vel_new
        posi_curr = posi_new
        total_reward = total_reward + reward_use

        #if car is getting stuck in a state then increase randomness
        if(total_reward==0):
            accumulate.append(reward_use)
        if(len(accumulate)>10):
            eps = 0.8
            accumulate.clear()

    #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py  
    if(i_episode%500==0):
        print('episode ', i_episode, 'score ', total_reward, 'epsilon %.3f' % eps)
        
    #epsilon decay strategy referred from
    #https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
    eps = eps - 2/10000 if eps > 0.01 else 0.01
    reward_plot.append(total_reward)
    if(i_episode==0):
        reward_mean.append(np.float(total_reward))
    elif(i_episode>0):
        reward_mean.append(np.float(reward_mean[-1]+total_reward))
    
    #terminate training if car reached goal 5 times
    if(flag==1):
        break
    
    
env.close()
mean = sum(reward_plot[25::])/((len(reward_plot)-25))
mean_print = []
for i in range(len(reward_mean)):
    mean_print.append((reward_mean[i])/(i+1))
#following 5 lines referred from
#https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
plt.plot(reward_plot,label = "Episode Reward")
plt.plot(mean_print,label = "Average Reward")
plt.legend(loc="best")
plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)
#following ine referred from
#https://stackoverflow.com/questions/52666450/put-text-label-at-the-end-of-every-line-plotted-through-matplotlib-with-three-di
plt.annotate(xy=(len(reward_plot),reward_plot[-1]), xytext=(5,0), textcoords='offset points' ,fontsize=14,text="Car reached \n the goal\n 10th time", va='center')
plt.show()