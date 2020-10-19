# hzy
# hzy
# 2020/10/18/19:12
# test

import numpy as np
import pandas as pd
import time
import pickle

#np.random.seed(2)

MAPHEIGHT = 5
MAPWEIGHT = 5
ACTIONS = ['left','right','up','down']

EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 13   # maximum episodes
FRESH_TIME = 0.3    # fresh time for one move


def getIndex(state):
    return (MAPWEIGHT*state[0,1] + state[0,0])

def build_q_table(env,action):
    table = pd.DataFrame(
        np.zeros((np.size(env),len(action))),
        columns = action,
        index = range(np.size(env))
    )
    return table

def choose_action(state,q_table):
    index = getIndex(state)
    state_actions = q_table.iloc[index,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()
    return action_name

def get_env_feedback(state,action,env):
    #ACTIONS = ['left', 'right', 'up', 'down']
    nextState = state.copy()
    if action == 'left':
        if state[0,0]== 0:
            nextState[0,0] = state[0,0]
        else:
            nextState[0,0] = state[0,0] - 1

        if env[nextState[0,0],nextState[0,0]] == 0:
            reward = 0
        else:
            reward = 1
            nextState = 'terminal'

    elif action == 'right':
        if state[0,0]== MAPWEIGHT-1:
            nextState[0,0] = state[0,0]
        else:
            nextState[0,0] = state[0,0] + 1

        if env[nextState[0,0],nextState[0,0]] == 0:
            reward = 0
        else:
            reward = 1
            nextState = 'terminal'

    elif action == 'up':
        if state[0,1] == 0:
            nextState[0,1] = state[0,1]
        else:
            nextState[0,1] = state[0,1] - 1

        if env[nextState[0,0],nextState[0,0]] == 0:
            reward = 0
        else:
            reward = 1
            nextState = 'terminal'

    #elif action == 'down':
    else :
        if state[0,1] == MAPHEIGHT - 1:
            nextState[0,1] = state[0,1]
        else:
            nextState[0,1] = state[0,1] + 1

        if env[nextState[0,0],nextState[0,0]] == 0:
            reward = 0
        else:
            reward = 1
            nextState = 'terminal'

    return nextState,reward

def update_env(state,env):
    if state == 'terminal':
        showEnv(state,env)
        time.sleep(2)
    else:
        showEnv(state, env)
        time.sleep(FRESH_TIME)

def showEnv(state,env):
    for iTemp in range(MAPWEIGHT):
        for jTemp in range(MAPHEIGHT):
            if state == 'terminal':
                if env[iTemp,jTemp] == 1:
                    print('T')
                else:
                    print('*',end='')
            else:
                if ((iTemp == state[0,0]) and (jTemp == state[0,1])):
                    print('o',end='')
                elif env[iTemp,jTemp] == 1:
                    print('T', end='')
                else:
                    print('*',end='')
        print('')

def rl(env,pre_table):
    # main part of RL loop
    q_table = pre_table.copy()
    for episode in range(MAX_EPISODES):
        state = np.zeros((1,2),dtype='int')
        is_terminated = False
        update_env(state, env)
        while not is_terminated:
            action = choose_action(state, q_table)
            nextState, reward = get_env_feedback(state,action,env)  # take action & get next state and reward
            q_predict = q_table.loc[getIndex(state), action]
            if nextState!= 'terminal':
                q_target = reward + GAMMA * q_table.iloc[getIndex(nextState), :].max()  # next state is not terminal
            else:
                q_target = reward  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.loc[getIndex(state),action] += ALPHA * (q_target - q_predict)  # update
            state = nextState  # move to next state

            update_env(state, env)
            print(q_table)
    return q_table

if __name__ == "__main__":
    try:
        fp = open('env.pickle','rb')
    except Exception as e:
        print(e)
        env = np.zeros((MAPWEIGHT,MAPHEIGHT))
        env[MAPWEIGHT-1,MAPHEIGHT-1] = 1
    else:
        fp.close()
        env = pickle.load(fp)


    try:
        fp = open('Qtable.pickle','rb')
    except Exception as e:
        print(e)
        pre_table = build_q_table(env,ACTIONS)
    else:
        fp.close()
        pre_table = pd.read_pickle('Qtable.pickle')

    q_table = rl(env,pre_table)
    print('\r\nQ-table:\n')
    print(q_table)

    try:
        fp = open('env.pickle','wb')
    except Exception as e:
        print(e)
        print("can't save the environment")
    else:
        pickle.dump(env, fp)
        fp.close()

    q_table.to_pickle('Qtable.pickle')
