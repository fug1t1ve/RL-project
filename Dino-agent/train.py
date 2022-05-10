import time
import pandas as pd
import numpy as np
from utils import *
from init import *
from tensorflow.keras.optimizers import Adam
import json
import random
from IPython.display import clear_output

def trainNetwork(model,game_state,observe=False):
    last_time = time.time()
    D = load_obj("D")
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] =1
    
    x_t, r_0, terminal = game_state.get_state(do_nothing) 

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) 

    
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) 
    initial_state = s_t 

    if observe :
        OBSERVE = 999999999   
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                      
        OBSERVE = OBSERVATION
        epsilon = load_obj("epsilon") 
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)

    t = load_obj("time")
    while (True): 
        
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0 
        a_t = np.zeros([ACTIONS]) 

        if t % FRAME_PER_ACTION == 0: 
            if  random.random() <= epsilon: 
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)       
                max_Q = np.argmax(q)        
                action_index = max_Q 
                a_t[action_index] = 1        
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

        x_t1, r_t, terminal = game_state.get_state(a_t)
        print('fps: {0}'.format(1 / (time.time()-last_time))) 
        last_time = time.time()
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) 

        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE: 

            minibatch = random.sample(D, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   
            targets = np.zeros((inputs.shape[0], ACTIONS))       
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]   
                action_t = minibatch[i][1]  
                reward_t = minibatch[i][2]   
                state_t1 = minibatch[i][3]   
                terminal = minibatch[i][4]   
                

                inputs[i:i + 1] = state_t    

                targets[i] = model.predict(state_t)  
                Q_sa = model.predict(state_t1)     
                
                if terminal:
                    targets[i, action_t] = reward_t 
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
        s_t = initial_state if terminal else s_t1 
        t = t + 1

        if t % 1000 == 0:
            print("Now we save model")
            game_state._game.pause() 
            model.save_weights("model.h5", overwrite=True)
            save_obj(D,"D") 
            save_obj(t,"time") 
            save_obj(epsilon,"epsilon") 
            loss_df.to_csv("./objects/loss_df.csv",index=False)
            scores_df.to_csv("./objects/scores_df.csv",index=False)
            actions_df.to_csv("./objects/actions_df.csv",index=False)
            q_values_df.to_csv(q_value_file_path,index=False)
            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)
            clear_output()
            game_state._game.resume()
        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")