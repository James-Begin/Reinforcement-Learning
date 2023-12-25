import argparse
import os

import datetime as dt
import numpy as np
import pandas as pd
import itertools
import pickle
import re

import keras.optimizers
from sklearn.preprocessing import StandardScaler
from keras.models import Model, Sequential
from keras.layers import Dense, Input


import sklearn.preprocessing

import yfinance as yf

def get_prices():

    #generate data
    start = dt.datetime(2015, 1, 1)
    end = dt.datetime(2020, 1, 1)
    SPY_data = yf.download('SPY', start, end)
    QQQ_data = yf.download('QQQ', start, end)
    Bond_data = yf.download('TLT', start, end)

    close_prices = pd.DataFrame()

    close_prices[0] = SPY_data['Adj Close']
    close_prices[1] = QQQ_data['Adj Close']
    close_prices[2] = Bond_data['Adj Close']
    close_prices.reset_index(drop=True, inplace=True)
    
    
    return close_prices.values

class ExperienceBuffer:
    #constructor that outlines the buffer
    def __init__(self, states, actions, size):
        self.action_buffer = np.zeros(size, dtype=np.uint8)
        self.reward_buffer = np.zeros(size)
        self.done_flag = np.zeros(size)
        self.state_buffer = np.zeros([size, states])
        self.next_state_buffer = np.zeros([size, states])
        #store buffers at pointer index
        self.pointer = 0
        self.max_size = size
        self.size = 0
    #
    def store(self, state, action, reward, next_state, done):
        self.action_buffer[self.pointer] = action
        self.state_buffer[self.pointer] = state
        self.next_state_buffer[self.pointer] = next_state
        self.reward_buffer[self.pointer] = reward
        self.done_flag[self.pointer] = done
        #once the entire buffer is full, start filling from the top again
        self.pointer = (self.pointer + 1) % self.max_size
        #check if buffer is full
        self.size = min(self.size+1, self.max_size)

    def sample(self, sample_size):
        index = np.random.randint(0, self.size, size = sample_size)
        return dict(s = self.state_buffer[index], s2 = self.next_state_buffer[index],
                    a = self.action_buffer[index], r = self.reward_buffer[index],
                    d = self.done_flag[index])


#create a sci kit scaler object for our data
#as we dont have a trained agent, randomly access episodes and record states to fit scaler
def scaler(environment):
    states = []
    for x in range(environment.days):
        action = np.random.choice(environment.action_space)
        state, reward, done, info = environment.step(action)
        states.append(state)
        if done:
            break
    scaler = StandardScaler()

    scaler.fit(states)

    return scaler

#create a neural network used by agent
def form_model(input_dimension, actions):
    model = Sequential()

    model.add(Dense(32, input_dim = input_dimension, activation = 'relu'))
    model.add(Dense(actions))

    model.compile(loss = 'mse', optimizer = 'adam')


    i = Input(shape=(input_dimension,))
    x = i

    x = Dense(32, activation='relu')(x)

    x = Dense(actions)(x)

    model = Model(i, x)
    model.compile(loss='mse', optimizer='adam')
    
    return model

#create environment of our chosen equities, this is where we complete our actions
class environment:
    def __init__(self, prices):
        self.data = prices
        self.days, self.num_stocks = self.data.shape

        #initialize needed vars
        self.principal = 10000
        self.owned = None
        self.price = None
        self.cash = None
        self.day = None

        #here we create a space for our actions, there are 3**3 permutations as there are 3 stocks and 3 actions (hold, sell, buy)
        self.action_space = np.arange(3**self.num_stocks)


        #create a map of the different actions:
        #for example: [1,0,2] (buy, hold, sell)
        self.actions = list(map(list, itertools.product([0,1,2], repeat=self.num_stocks)))

        self.state_dimensions = self.num_stocks * 2 + 1

        self.reset()

    def reset(self):
        self.day = 0
        self.owned = np.zeros(self.num_stocks)
        self.price = self.data[self.day]
        self.cash = self.principal

        return self.get_state()

    def step(self, action):
        assert action in self.action_space

        previous_value = self.get_value()

        #increment and update price
        self.price = self.data[self.day]
        self.day += 1


        self.trade(action)

        current_value = self.get_value()

        current_reward = current_value - previous_value

        #if all data has been used, done flag
        done = self.day = self.days - 1

        track_value = {'current_value':current_value}

        return self.get_state(), current_reward, done, track_value

    #return current portfolio value
    def get_value(self):
        sum = 0
        for i in range(self.num_stocks):
            sum += self.owned[i] + self.price[i]
        return sum + self.cash

    #here we set the state, which contains the stock owned, # of stocks, and cash in a list
    def get_state(self):
        action_store = np.empty(self.state_dimensions)
        #we are only observing 3 equities, so these would create lists of 3
        action_store[:self.num_stocks] = self.owned
        action_store[self.num_stocks:2 * self.num_stocks] = self.price

        action_store[-1] = self.cash

        return action_store

    def trade(self, trade):
        #first get the trades we want to perform
        trades = self.actions[trade]

        to_sell = []
        to_buy = []

        for stock, action in enumerate(trades):
            #determine what kind of trades
            if action == 0:
                to_sell.append(stock)
            elif action == 2:
                to_buy.append(stock)


        #sell first to ensure we have enough cash
        if to_sell:
            for stock in to_sell:

                #sell all shares and add to cash
                self.cash += self.price[stock] * self.owned[stock]
                self.owned[stock] = 0
        if to_buy:

            #buy with an equal weighting
            self.temp_cash = self.cash / len(to_buy)
            for stock in to_buy:
                if self.temp_cash > self.price[stock]:
                    self.owned[stock] += (self.temp_cash // self.price[stock])
                    self.cash -= self.owned[stock] * self.price[stock]

#The Reinforcement Learning portion
class Agent(object):
    #initialize needed values and hyperparameters (to be tweaked)
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        #create a buffer with a size of 500, for now, this is arbitrary
        self.experience_buffer = ExperienceBuffer(state_size, action_size, size = 500)
        #the value of future rewards (i.e. future rewards are discounted against current)
        self.discount = 0.95
        #this is the epsilon in the epsilon greedy algorithm
        #essentially the probability of randomly choosing an action (exploring)
        self.explore_rate = 1.0
        self.min_explore = 0.01
        #rate at which the exploration rate changes
        self.explore_delta = 0.995

        self.model = form_model(state_size, action_size)

    #choose an action to perform
    def action(self, state):
        #randomly perform an action (epsilon-greedy)
        if np.random.rand() < self.explore_rate:
            return np.random.choice(self.action_size)
        #perform a "greedy" action and return maximum reward
        action = self.model.predict(state)
        return np.argmax(action[0])

    def update_buffer(self, s, a, r, s2, d):
        self.experience_buffer.store(s,a,r,s2,d)

    def experience_replay(self,size = 32):
        #sample size

        #return if not buffer is not filled enough
        if self.experience_buffer.size < size:
            return

        #sample from the buffer
        batch = self.experience_buffer.sample(size)
        s = batch['s']
        a = batch['a']
        r = batch['r']
        s2 = batch['s2']
        d = batch['d']

        #calculate targets in order to train our network
        target = r + (1-d) * self.explore_rate * np.amax(self.model.predict(s2), axis=1)

        #(target)
        target_update = self.model.predict(s)

        target_update[np.arange(size), a] = target

        #train once
        self.model.train_on_batch(s, target_update)

        #update exploration rate as it decreases over time
        if self.explore_rate > self.min_explore:
            self.explore_rate *= self.explore_delta

    #instead of retraining the model every run, train once, then save and load weights when needed
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play(agent, environment, training):

    s = environment.reset()

    s = Scaler.transform([s])

    done = False

    while not done:
        #determine next action
        a = agent.action(s)
        #perform action
        s2, r, d, track_value = environment.step(a)

        s2 = Scaler.transform([s2])

        #check if currently training
        if training == 'train':
            #update buffer and train once
            agent.update_buffer(s, a, r, s2, d)
            agent.experience_replay()

        #increment state
        s = s2

    return track_value['current_value']


if __name__ == '__main__':

    #number of episodes to run through
    num_iter = 1000
    #size of batch drawn from buffer
    size = 32

    #here we can provide a way to choose whether to test the model or train the model from the cmd line
    #https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    #choose either training or testing
    parser.add_argument('-m', '--mode', required=True, type=str, help = 'Choose train or test')
    args = parser.parse_args()

    #create folders to store the model and rewards
    if not os.path.exists('trading_models'):
        os.makedirs('trading_models')
    if not os.path.exists('trading_rewards'):
        os.makedirs('trading_rewards')

    p = get_prices()
    #train on first half of the data and test on the rest
    num_train = p.shape[0] // 2
    training_data = p[:num_train]
    testing_data = p[num_train:]

    #create instance using training data
    #init agent using the size of our states and actions
    env = environment(training_data)
    states_dim = env.state_dimensions
    actions_dim = len(env.action_space)

    agent = Agent(states_dim, actions_dim)
    Scaler = scaler(env)

    #track total value
    value = []

    if args.mode == 'test':
        #load the previously trained model
        with open(f'{"trading_models"}/scaler.pkl', 'rb') as x:
            Scaler = pickle.load(x)

        env = environment(testing_data)

        #exploration rate should be between 0 and 1 exclusive
        agent.explore_rate = 0.01
        #pass in model
        agent.load(f'{"trading_models"}/dqn.h5')

    #run episodes
    for i in range(num_iter):
        t = dt.datetime.now()
        curr_val = play(agent, env, args.mode)
        elapsed = dt.datetime.now() - t
        print(f"Iteration: {i + 1}/{num_iter}, End Value: {curr_val:.2f}, Time Elapsed: {elapsed}")
        value.append(curr_val)

    if args.mode == 'train':
        #first save model
        agent.save(f'{"training_models"}/dqn.h5')

        with open(f'{"training_models"}/scaler.pkl', 'wb') as x:
            pickle.dump(scaler, x)

    np.save(f'{"training_rewards"}/{args.mode}.npy', value)
