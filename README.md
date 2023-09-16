# Reinforcement-Learning
Reinforcement Learning is a significant subfield of Machine Learning and is well known for being able to dominate in games of strategy like Go and Chess. RL involves an agent interacting in an environment similar to a player in a game of chess or a driver on the road. When an agent performs actions in its environment it either recieves a reward or a penalty. The goal of the agent is to learn a "policy" or a plan for what action to take depending on the current state to maximize total rewards over time. This kind of model can be incredeibly powerful in skilled games that involve decisions over time with distinct rewards and penalties. For example, in Chess, it is not very difficult to identify a good move and a poor move that the model can learn from. Further, one clear application for RL is in trading due to its similarity to a game of skill such as chess. In trading, there are clear rewards, positive returns, and penalties, negative returns.

There are five key components in RL:  
**Agent:** The entity that is learning and making decisions. In this case, the agent is the model deciding to make trades  
**Environment:** The context which the agent operates in. For us, this is the market made up of a select few equities.  
**States:** The conditions or positions the agent can be in. This can include the price of the shares, how many we own, and the amount of cash on hand.  
**Actions:** The possible decisions the agent can make. This is simply whether to buy, sell, or hold our shares based on the environment.
**Rewards:** The feedback the agent receives after performing an action. In our case, this is whether we take a profit or a loss when trading. 
