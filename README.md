# Reinforcement-Learning
Reinforcement Learning is a significant subfield of Machine Learning and is well known for being able to dominate in games of strategy like Go and Chess. RL involves an agent interacting in an environment similar to a player in a game of chess or a driver on the road. When an agent performs actions in its environment it either recieves a reward or a penalty. The goal of the agent is to learn a "policy" or a plan for what action to take depending on the current state to maximize total rewards over time. This kind of model can be incredeibly powerful in skilled games that involve decisions over time with distinct rewards and penalties. For example, in Chess, it is not very difficult to identify a good move and a poor move that the model can learn from. Further, one clear application for RL is in trading due to its similarity to a game of skill such as chess. In trading, there are clear rewards, positive returns, and penalties, negative returns.

There are five key components in RL:  
  
**Agent:** The entity that is learning and making decisions. In this case, the agent is the model deciding to make trades  
  
**Environment:** The context which the agent operates in. For us, this is the market made up of a select few equities.  
  
**States:** The conditions or positions the agent can be in. This can include the price of the shares, how many we own, and the amount of cash on hand. 
  
**Actions:** The possible decisions the agent can make. This is simply whether to buy, sell, or hold our shares based on the environment.  
  
**Rewards:** The feedback the agent receives after performing an action. In our case, this is whether we take a profit or a loss when trading. 
  
## Agent
The agent is defined in the "Agent" class:

```
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
```
Here we initialize needed values like the exploration rate and discount rate. These values help to determine how quickly and randomly our agent acts. In the agents class, we also define the action function: 
```
    def action(self, state):
        #randomly perform an action (epsilon-greedy)
        if np.random.rand() < self.explore_rate:
            return np.random.choice(self.action_size)
        #perform a "greedy" action and return maximum reward
        action = self.model.predict(state)
        return np.argmax(action[0])
```
This simply chooses between a random action or a reward maximizing action based on the explore rate. Next is the replay and buffer:
```
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
```
The experience replay is a place to store the agent's actions at each point in time. This way, we can reuse previous experiences for training, helping to make the learning process more efficient. By sampling previous experiences, we can prevent the effects of correlated consecutive data, further helping to stabilize the learning process. At the end of the class, we define store and load functions to prevent having to retrain every time:
```
#instead of retraining the model every run, train once, then save and load weights when needed
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
```
## Environment
In this case, the environment is the stock market (made up of only provided data) and is defined in the environment class:
```
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
```
The environment is initialized with the stock pricing data, and all the necessary variables like owned stocks and amount of cash. Additionally, we also initialize the action space, which stores all possible actions based on the different stocks and actions. This action space is then used in our agent class. 
```
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
```
Next, the reset function simply initializes our state and stock prices at time=0. The step function updates the time and prices of stocks, then undergoes and action provided by the agent. Then, it returns the new state and rewards.
```

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
```
The two get functions allow for easy access to total portfolio value and state.
```
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
```
The trade function performs trades based on the provided actions from the agent. Overall, the function updates the holdings and ensures that there is enough capital to perform the required trades.
## Experience Buffer
The experience buffer is stored in the class with the same name and acts as a memory for past experiences:
```
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
```
The buffer stores, actions, rewards, state, next state, and a done flag. First all buffers are initialized and are filled in the store function. Further, the buffer has a maximum size and once it's filled, will start to be overwritten from the top. This helps prevent learning from past experiences too many times. Finally, the sample function provides a random sample of experiences drawn from the buffer. In this case, the sample size is arbitrary and set to 32.

# Results
![Rewardsplot](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/03812c1b-216e-41c7-8dd0-62bdf3b7a671)  
On average the model's average portfolio value was $20869.36 (108.7% return) with a minimum value of $15492.87 (54.9% return) and a maximum of $29895.43 (199.0% return).  
![sp500plot](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/08efd316-6ae3-4928-93b7-ba56a9f2a47d)  
In comparison, the S&P500 return 72.9% and the NASDAQ100 returned 116.5% over the same 5 year period.  
## Scenario 2: Increased Volatility
It is relatively simple to earn a profit when the markets consistently rise year over year. During the period of 2002 to 2012 the S&P 500 and NASDAQ 100 maintained an average annualized volatility of 21 to 24%, double their historical averages:  
![spyqqq20022012](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/d7e4fafb-7aa4-4e77-ae65-dd46dd766b2d)  
During this period, the S&P 500 returned 102.2% and the NASDAQ 100 returned 222.0%, significantly more than in the first example.  
![rewardsplot2](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/fcf25cbf-1fe3-489d-a2bd-d0176572664b)  
On average the model's average portfolio value was $18754.25 (84.5% return) with a minimum value of $8833.99 (-11.7% return) and a maximum of $30789.12 (207.9% return). During this period of increased volatility, its clear that the model vastly underperforms on average and even produces a loss in the worst case.  
## Scenario 3: Negative Returns
This scenario provides both negative returns and high volatility during the period 2004 to 2009 including the 2008 financial crisis:  
![20042009plot](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/96b0c2e2-1411-4409-b98c-2e946f126bd9)  
![scenario3plot](https://github.com/James-Begin/Reinforcement-Learning/assets/103123677/ebec2e8d-c60a-42f6-8606-239435c157fa)


