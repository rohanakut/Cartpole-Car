import matplotlib.pyplot as plt
import glob
import numpy as np


reward_batch = []
total_goal_load = []
flag=0

##read the npy files for episode wise progression
for i in range(0,3):
    if(flag==1):
        break
    for j in range(0,3):
        try:
            ##remove the comment tag("#") on line 17 and add a comment tag("#") on line 18 to point to 'results_done' folder
            #name_progress = './results_done/pregress_np'+str(i)+str(j)+'.npy'
            name_progress = './results/pregress_np'+str(i)+str(j)+'.npy'
            reward_batch.append(np.load((name_progress)))
        except Exception as e:
            print("file does not exist")
            flag=1
            break

##read the npy files for number of goals reached
for i in range(0,3):
    try:
        ##remove the comment tag("#") on line 29 and add a comment tag("#") on line 30 to point to 'results_done' folder
        #name = './results_done/result_np'+str(i)+'.npy'
        name = './results/result_np'+str(i)+'.npy'
        #https://stackoverflow.com/questions/28439701/how-to-save-and-load-numpy-array-data-properly
        total_goal_load.append(np.load((name),allow_pickle=True))
        #print(total_goal_load[-1])
    except Exception as e:
        print(e)
        print("file does not exist")
        break


goal_list = []
total_goal = []

##processing of the loaded values to make them according to matplotlib format
for i in range(0,3):
    for j in range(len(total_goal_load[i])):
        try:
            if(total_goal_load[i][j]==0):
                total_goal_load[i][j]=[0]
        except Exception as e:
            pass
        if(len(total_goal_load[i][j])==1):
            if(total_goal_load[i][j]==[0]):
                goal_list.append([])
                total_goal.append(0)
            else:
                #https://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-int
                goal_list.append([int(num) for num in total_goal_load[i][j]])
                goal_list[-1][-1] = goal_list[-1][-1] - 1
                total_goal.append(1)
        else:
            #https://stackoverflow.com/questions/7368789/convert-all-strings-in-a-list-to-int
            goal_list.append([int(num) for num in total_goal_load[i][j]])
            goal_list[-1][-1] = goal_list[-1][-1] - 1
            total_goal.append(len(total_goal_load[i][j]))


print(len(reward_batch[0]))
print(len(goal_list))
print(goal_list)
print(total_goal)
fig, axs = plt.subplots(3)

##following 2 lines referred from
#https://stackoverflow.com/questions/31006971/setting-the-same-axis-limits-for-all-subplots-in-matplotlib
custom_xlim = (0, 1000)
plt.setp(axs, xlim=custom_xlim)

##all the following lines are referred from
#https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
axs[0].plot(list(reward_batch[0][0]),'-D',markevery=(goal_list[0]),label=("Batch Size = 30 \nGoal reached ="+str((total_goal[0]))))
axs[0].plot(list(reward_batch[1][0]),'-D',markevery=(goal_list[1]),label=("Batch Size = 100 \nGoal reached ="+str((total_goal[1]))))
if(len(reward_batch[2][0])>5):
    axs[0].plot(list(reward_batch[2][0]),'-D',markevery=list(goal_list[2]),label=("Batch Size = 500 \nGoal reached ="+str((total_goal[2]))))
axs[0].set_title('Learning Rate = 0.01')
plt.legend()
axs[1].plot(list(reward_batch[3][0]),'-D',markevery=list(goal_list[3]),label=("Batch Size = 30 \nGoal reached ="+str((total_goal[3]))))
axs[1].plot(list(reward_batch[4][0]),'-D',markevery=list(goal_list[4]),label=("Batch Size = 100 \nGoal reached ="+str((total_goal[4]))))
if(len(reward_batch[5][0])>5):
    axs[1].plot(list(reward_batch[5][0]),'-D',markevery=list(goal_list[5]),label=("Batch Size = 500 \nGoal reached ="+str((total_goal[5]))))
axs[1].set_title('Learning Rate = 0.001')
plt.legend()
axs[2].plot(list(reward_batch[6][0]),'-D',markevery=list(goal_list[6]),label=("Batch Size = 30 \nGoal reached ="+str((total_goal[6]))))
axs[2].plot(list(reward_batch[7][0]),'-D',markevery=list(goal_list[7]),label=("Batch Size = 100 \nGoal reached ="+str((total_goal[7]))))
if(len(reward_batch[8][0])>5):
    axs[2].plot(list(reward_batch[8][0]),'-D',markevery=list(goal_list[8]),label=("Batch Size = 500 \nGoal reached ="+str((total_goal[8]))))
axs[2].set_title('Learning Rate = 0.0001')
plt.legend()
lines = []
labels = []
plt.tight_layout()

##following 3 lines referred from
#https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib
axs[0].legend(title="Batch Size and Goal Reached",fontsize=10,title_fontsize=10,loc="lower right",bbox_to_anchor=(1.0, 0),ncol=2)
axs[1].legend(title="Batch Size and Goal Reached",fontsize=10,title_fontsize=10,loc="lower right",bbox_to_anchor=(1.0, 0),ncol=2)
axs[2].legend(title="Batch Size and Goal Reached",fontsize=10,title_fontsize=10,loc="lower right",bbox_to_anchor=(1.0, 0),ncol=2)

plt.xlabel("Episodes",fontsize=18)
plt.ylabel("Reward",fontsize=18)
plt.show()