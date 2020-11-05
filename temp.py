
###Imports### 1
import timeit
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras
#Using TensorFlow backend.

###Aufbau des Models### 2
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 1.0   # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Einfaches NN 
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu', 
                        kernel_regularizer=keras.regularizers.l2(0.00001)))
        #model.add(Dropout(0.3))
        #model.add(Dense(24, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done, total, importance):
        # merkt sich alle bisher durchlaufenen Zustände
        self.memory.append([state, action, reward, next_state, done,total,importance])

    def act(self, state):
        # epsilon-greedy: off-policy oder policy
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        # baut den Vektor der Q-Werte aus 
        # als reward zum Zeitpunkt t + gamma*max(moegliche rewards zum Zeitpunkt t+1)
        
        probabilities = np.array([m[-1] for m in self.memory])
        probabilities = 1./np.sum(probabilities) * probabilities
        #print( probabilities.shape)
        minibatch = [self.memory[i] for i in np.random.choice(range(len(self.memory)),size=batch_size, p=probabilities)]
        states, targets_f = [], []
        for state, action, reward, next_state, done,total,importance in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
        
        
###Trainig### 3
        
EPISODES = 22


env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32

for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    cum_reward = 0
    for time in range(500):
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        additional_reward = -(state[0,0] + state[0,0]*state[0,2]-state[0,1]*state[0,3])##faktore aus probieren
        reward = reward + additional_reward if not done else 10 #
        cum_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done,reward,1)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}"
                  .format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
            # Logging training loss and actual reward every 10 timesteps
            if time % 10 == 0:
                print("episode: {}/{}, time: {}, cumulative reward: {:.4f}, loss: {:.4f}".format(e, EPISODES, time, cum_reward, loss)) 
        
    
    for i in range(time):
        pos = -i-1
        agent.memory[-i-2][-2] += reward
        for j in range(-time,pos):
            new_total =  agent.memory[j][-2] + agent.memory[pos][2]
            mem = agent.memory[j]
            agent.memory[j][-1] =new_total

    for i in range(time):
        pos = -i-1
        imp = max(agent.memory[pos][-2]-agent.model.predict(agent.memory[pos][0])[0,agent.memory[pos][1]],0)
        mem = agent.memory[pos]
        agent.memory[pos][-1] = imp
            
            
    agent.save("qlearning_Acrobot_3versuche")
    
    
    
###Testen### 4
    
    
import gym
env = gym.make('Acrobot-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
done = False
batch_size = 32
zähler=0

agent.load("qlearning_Acrobot_3versuche")

import time  as ti
for e in range(100):
    state = env.reset()
    #state[0] = state[0] + np.random.randn()*0.1
    #state[1] = state[1] + np.random.randn()*0.1
    #state[2] = state[2] + np.random.randn()*0.1
    #state[3] = state[3] + np.random.randn()*0.1
    #env.env.state = state
    state = np.reshape(state, [1, state_size])
    for time in range(2000):
        
        env.render()#display funktion geht bei google nicht
        agent.epsilon = 0
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        if done:
            zähler+=1
            print (zähler,   "Duration: ", time)
            break
            
    else:
        print ("Volle Zeit")
        
###zusammenfassung### 5
agent.model.summary()
