##the import "neuralNet" is referred from 
##https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
import neuralNet

import os
import gym
import numpy as np
import random
import ipdb
import math
import matplotlib.pyplot as plt
env = gym.make('MountainCar-v0')

#increase length of episode. Referred from
#https://www.reddit.com/r/reinforcementlearning/comments/agd6j4/how_to_change_episode_length_and_reward/
env._max_episode_steps = 2000

#entire function referred from
#https://www.youtube.com/watch?v=KzsBaqYzNLc&ab_channel=Jakester897
def get_reward(state):
    if(state>=0.5):
        return 2
    else:
        reward = (state+1.2)/1.8 -1
    return reward

replay_mem = []

## initialise the replay memory with 20,000 random samples
def generate_replay():
    for i in range(100):
        observation = env.reset()
        for t in range(200):
            #ipdb.set_trace()
            state1 = observation[0]
            state2 = observation[1]
            action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
            new_state = [new_observation[0],new_observation[1]]
            reward = get_reward(new_observation[0])
            attach = ([state1,state2],reward,new_state,action)
            #print(attach)
            replay_mem.append(attach)
            if(done):
                break
            observation = new_observation
    return replay_mem

##create a directory to save results
##https://www.tutorialspoint.com/How-can-I-create-a-directory-if-it-does-not-exist-using-Python
if not os.path.exists('results'):
    os.makedirs('results')

start_lr=0
start_batch=0
reward_batch = []
total_goal = []
#read_text('./progress.txt')
replay_mem = generate_replay()
reward_tot = []
print(len(replay_mem))

eps = 1.0

#declare neural network architecture
layers = [2,150,100,3]

##initialise 2 neural networks
param = neuralNet.initialize_params(layers)
base_param = neuralNet.initialize_params(layers)


flag=0
goal = 0
flag_goal = 0
goal_list = []


for i in range(len(total_goal)):
    if(total_goal[i]==0):
        goal_list.append(list([]))
    else:
        goal_list.append(total_goal[i])

##iterate over different batch and learning rate values
batch = [30,100,500]
lr = [0.01,0.001,0.0001]

for lr_i in range(len(lr)):
    for batch_i in range(len(batch)):
        reward_mean = []
        reward_batch = []
        loss= 0
        eps = 1.0
        goal_lis = []
        flag=0
        goal = 0
        param = neuralNet.initialize_params(layers)
        base_param = neuralNet.initialize_params(layers)
        for i in range(1000):
            observation = env.reset()
            tot_reward = 0
            for t in range(1000):
                #ipdb.set_trace()    
                state1 = observation[0]
                state2 = observation[1]
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
                
                ##get custom reward
                reward = get_reward(new_observation[0])
                tot_reward = tot_reward + reward

                ##termination condition to consider that MountainCar is solved
                if(new_observation[0]>=0.5):
                    if(eps<0.5):
                        goal = goal+1
                        goal_lis.append(i)
                    print("Goal is:",goal)
                    print("Episode is:",i)
                    done = True
                    #terminate if car reached goal 3 times
                    if(goal >= 3):
                        flag = 1
                    break
                new_state1 = new_observation[0]
                new_state2 = new_observation[1]
                
                ##append to replay memory and pop the first sample
                if(len(replay_mem)>2000):
                    replay_mem.pop(0)
                replay_mem.append(([state1,state2],reward,[new_state1,new_state2],action))
                
                if(done):
                    break
                
                ##randomly sample from replay memory. 
                ##following line referred from
                ##https://stackoverflow.com/questions/22842289/generate-n-unique-random-numbers-within-a-range
                sample = random.sample(range(0, len(replay_mem)), batch[batch_i])
                
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
                next_state_y = neuralNet.forward_propagation(np.transpose(input_next_state),base_param)
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
                y = reward + 0.99*max_next_state

                ##processing to compute loss. I have used L1 loss in this code
                y_final = y_pred.copy()
                y_pred_final = []
                for i_curr in range(len(y_final)):
                    y_pred_final.append(y_final[i_curr][action[i_curr]])
                    y_final[i_curr][action[i_curr]] = y[i_curr]
                try:
                    loss = np.mean(np.abs(y-y_pred_final))
                except Exception as e:
                    print("Exception ocurred")
                
                ##perform gradient descent
                ##https://towardsdatascience.com/an-introduction-to-neural-networks-with-implementation-from-scratch-using-python-da4b6a45c05b
                grads = neuralNet.backward_propagation(param,y_pred_dict,np.transpose(input_curr_state),np.transpose(y_final))
                ##update parameters
                param = neuralNet.update_params(param, grads, lr[lr_i])
                observation = new_observation

            ##epsilon decay
            ##https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/Fundamentals/mountaincar.py
            eps = eps - 2/100 if eps > 0.01 else 0.01
    
            if(i%100==0):
                print(i,eps,loss,tot_reward)
            if(math.isnan(loss)):
                print(eps)
                print("Gradient Exploded")
                break
            reward_tot.append(tot_reward)

            ##update target network every 1000 episodes. This is done to keep the DQN stable
            if(i%100==0 and i>100):
                base_param = param

            if(flag==1):
                break

            if(i==0):
                reward_mean.append(np.float(tot_reward))
            else:
                reward_mean.append(np.float(reward_mean[-1]+tot_reward))
        mean_print = []
        for i in range(len(reward_mean)):
            if(math.isnan(reward_mean[i])):
                break
            mean_print.append((reward_mean[i])/(i+1))
        reward_mean_copy = mean_print.copy()
        reward_batch.append(reward_mean_copy)
        if(len(goal_lis)==0):
            total_goal.append([0])
        else:
            total_goal.append(goal_lis)
       
        goal_list.append(goal_lis)
        ##following 2 lines referred from
        ##https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
        np.save(('./results/result_np'+str(lr_i)),np.array(total_goal))
        np.save(('./results/pregress_np'+str(lr_i)+str(batch_i)),np.array(reward_batch))
        
        print("Batch finished",batch_i)
    start_batch =0
    print("Alpha finished",lr_i)
print("Finished!!!")
print(len(total_goal))
print(total_goal[0])
