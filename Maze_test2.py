# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:24:28 2020

@author: AntonioTseng
"""

from medium_qlearning_env import Env
import numpy as np
import time
import os

# create environment
env = Env()

# QTable : contains the Q-Values for every (state,action) pair
qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

# hyperparameters
epochs = 250
#r 折扣因子
gamma = 0.1
#E
epsilon = 0.08
#衰變因子
decay = 0.1
Tsteps=[]
# training loop
for i in range(epochs):
    state, reward, done = env.reset()
    steps = 1
    
    while not done:
        #print('')
        os.system('clear')

        
        print("epoch #", i+1, "/", epochs)
        print("steps : ",steps)
# =============================================================================
#         if action == 0:
#             print("要做動作: 左")
#         elif action == 1:
#             print("要做動作: 右")
#         elif action == 2:
#             print("要做動作: 上")
#         elif action == 3:
#             print("要做動作: 下")
#         elif action == 4:
#             print("要做動作: 接人")
# =============================================================================
# =============================================================================
#         a=0
#         for x in qtable:
#             a+=1
#             max_x=max(x)
#             if x.index(max_x) == 0:
#                 print("state ",a,"",x,"",max_x,"",x.index(max_x)+1,"要做動作: 左")
#             elif x.index(max_x) == 1:
#                 print("state ",a,"",x,"",max_x,"",x.index(max_x)+1,"要做動作: 右")
#             elif x.index(max_x) == 2:
#                 print("state ",a,"",x,"",max_x,"",x.index(max_x)+1,"要做動作: 上")
#             elif x.index(max_x) == 3:
#                 print("state ",a,"",x,"",max_x,"",x.index(max_x)+1,"要做動作: 下")
# =============================================================================
        
        env.render()
        
        print("得到分數 : ",reward)
        
        #時間間格
        #time.sleep(0.01)

        # count steps to finish game
        steps += 1
            
        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.randomAction()
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))

        # 行動選擇(take action)
        next_state, reward, done = env.step(action)

        # Q Table 的更新方式，貝爾曼方程式(update qtable value with Bellman equation)
        qtable[state][action] = reward + gamma * max(qtable[next_state])
        #%%
# =============================================================================
#         print("qtable[next_state] : ",qtable[next_state])
#         print("qtable[state][action] : ",qtable[state][action])
#         print("len(qtable) : ",len(qtable))
#         print("qtable[state][action] : ",qtable)
# =============================================================================
        #%%
# =============================================================================
#         a=0
#         for i in qtable:
#             a+=1
#             max_i=max(i)
#             if i.index(max_i) == 0:
#                 print("state ",a,"",i,"",max_i,"",i.index(max_i)+1,"要做動作: 左")
#             elif i.index(max_i) == 1:
#                 print("state ",a,"",i,"",max_i,"",i.index(max_i)+1,"要做動作: 右")
#             elif i.index(max_i) == 2:
#                 print("state ",a,"",i,"",max_i,"",i.index(max_i)+1,"要做動作: 上")
#             elif i.index(max_i) == 3:
#                 print("state ",a,"",i,"",max_i,"",i.index(max_i)+1,"要做動作: 下")
#             #print("state ",a,"",i,"",max_i,"",i.index(max_i)+1)
# =============================================================================
        #%%
        # update state
        state = next_state
    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon

    print("\nDone in", steps, "steps".format(steps))
    Tsteps.append(steps)
    print("Total steps: ",Tsteps)
    data=open (r'C:\Users\AntonioTseng\Desktop\data.txt','w+') 
    print(qtable ,file = data)
    data.close()

    #time.sleep(0.5)