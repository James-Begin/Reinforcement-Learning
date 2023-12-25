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
`
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

