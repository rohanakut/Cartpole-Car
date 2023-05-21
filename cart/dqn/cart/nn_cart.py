#the import "neuralNet" is referred from
##https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
import neuralNet

import gym
import numpy as np
import random
import ipdb
import math
import matplotlib.pyplot as plt
env = gym.make('CartPole-v1')


replay_mem = []

## initialise the replay memory with 20,000 random samples
for i in range(100):
    observation = env.reset()
    for t in range(200):
        #ipdb.set_trace()
        state1 = observation[2]
        state2 = observation[3]
        action = env.action_space.sample()
        new_observation, reward, done, info = env.step(action)
        if(done and t<30):
        # print(t)
            reward = -1
        new_state = [new_observation[2],new_observation[3]]
        attach = ([state1,state2],reward,new_state,action)
        replay_mem.append(attach)
        if(done):
            break
        observation = new_observation

reward_tot = []
print("Length of replay memory is: ",len(replay_mem))
eps = 1.0
##define the neural network
layers = [2,150,100,2]
##initialise the neural network
param = neuralNet.initialize_params(layers)
#base_param = neuralNet.initialize_params(layers)


for i in range(7000):
    observation = env.reset()
    for t in range(200):
        #ipdb.set_trace()    
        state1 = observation[2]
        state2 = observation[3]
        curr_state = np.reshape(np.array([state1,state2]),(1,2))

        #get q value for current state
        q_val_curr = neuralNet.forward_propagation(np.transpose(curr_state),param)
        q_val_curr = q_val_curr["A3"]

        ##epsilon greedy strategy
        ##following 4 lines referred from 
        ##https://github.com/ikvibhav/reinforcement_learning/blob/master/code/mountain_car/mountain_car_sarsa.py
        if np.random.random() > eps:
            action = np.argmax(q_val_curr)
        else:
            action = np.random.randint(0, env.action_space.n)
        
        new_observation, reward, done, info = env.step(action)

        #following 2 lines are referred from
        ##https://github.com/JackFurby/CartPole-v0/blob/master/cartPole.py
        if(done and t<30):
            reward = -1
        
        new_state1 = new_observation[2]
        new_state2 = new_observation[3]

        ##append to replay memory and pop the first sample
        if(len(replay_mem)>2000):
            replay_mem.pop(0)
        replay_mem.append(([state1,state2],reward,[new_state1,new_state2],action))

        if(done):
            break

        ##randomly sample from replay memory. 
        ##following line referred from
        ##https://stackoverflow.com/questions/22842289/generate-n-unique-random-numbers-within-a-range
        sample = random.sample(range(0, len(replay_mem)), 30)

        input_curr_state = []
        input_next_state = []
        action = []
        reward = []

        ##processing after getting the samples
        for i_curr in range(len(sample)):
            input_curr_state.append(replay_mem[sample[i_curr]][0])
            input_next_state.append(replay_mem[sample[i_curr]][2])
            action.append(int(replay_mem[sample[i_curr]][3]))
            reward.append(replay_mem[sample[i_curr]][1])
        #ipdb.set_trace()
        input_curr_state = np.array(input_curr_state)
        input_next_state = np.array(input_next_state)

        #get q value for current state and next state
        y_pred_dict = neuralNet.forward_propagation(np.transpose(input_curr_state),param)
        y_pred = (np.transpose(y_pred_dict["A3"]))
        next_state_y = neuralNet.forward_propagation(np.transpose(input_next_state),param)
        next_state_y = (np.transpose(next_state_y["A3"]))
        #get the maximum q value for next state
        max_next_state = np.max(next_state_y,axis=1)

        ##determine if next state is terminal state. 
        # This is done by looking at reward function. 
        # If it is negative then state was terminal.
        for i_curr in range(len(max_next_state)):
            ##if next state is terminal state then do not consider it.
            if(reward[i_curr]==-1):
                max_next_state[i_curr] = 0

        #calculating the expected q value
        y = reward + 0.9*max_next_state

        #standard processing to calculate the loss. For my code I have used L1 loss
        y_final = y_pred.copy()
        y_pred_final = []
        for i_curr in range(len(y_final)):
            y_pred_final.append(y_final[i_curr][action[i_curr]])
            y_final[i_curr][action[i_curr]] = y[i_curr]
        loss = np.mean(np.abs(y-y_pred_final))

        ##perform gradient descent
        ##https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
        grads = neuralNet.backward_propagation(param,y_pred_dict,np.transpose(input_curr_state),np.transpose(y_final))
        #update parameters
        param = neuralNet.update_params(param, grads, 0.001)
        observation = new_observation
    
    ##epsilon decay
    ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
    eps = eps - 2/7000 if eps > 0.01 else 0.01
    
    if(i%1000==0):
        print(i,eps,loss)
    if(math.isnan(loss)):
        print(eps)
        break
    reward_tot.append(t)

    


print("Finished!!!")
mean_print = []
for i in range(len(reward_tot)):
    if(i==0):
        sum_reward = reward_tot[0]
    else:
        sum_reward = reward_tot[i] + sum_reward
    mean_print.append((sum_reward/(i+1)))

##following 5 lines are referred from 
#https://www.w3resource.com/graphics/matplotlib/basic/matplotlib-basic-exercise-5.php
plt.plot(reward_tot,label = "Episode Reward")
plt.plot(mean_print,label = "Average Reward")
plt.legend(loc="best")
plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)

plt.show()