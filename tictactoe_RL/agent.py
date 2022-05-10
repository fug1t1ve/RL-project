
from collections import defaultdict
import random
from env import *

class TabularAgent(object):
    def __init__(self, num_actions, alpha=0.75, gamma=1.0, epsilon=1.0, default_Q=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.default_Q = default_Q
        self.num_actions = num_actions  # should be 9 (9 squares so 9 possible actions)
        self.Q_tables = [{} for _ in range(self.num_actions)]
        
    # For a given board, return the action# that predicts the highest Q
    def max_action(self, state):
        # type: (BoardState) -> int
        predictions = [self.Q_read(ndx, state) for ndx in range(self.num_actions)]
        return np.argmax(predictions)

    # Choose a random action (0-8) with probability epsilon, 
    # or the optimal action with probability 1-epsilon
    def choose_action(self, state):
        # type: (BoardState) -> int
        if random.random() > self.epsilon:
            return random.choice(range(self.num_actions))
        return self.max_action(state)

    # Get Q-value for a particular action on a given board(-state)
    def Q_read(self, nAction, state):
        # type: (int, BoardState) -> float
        return self.Q_tables[nAction].get(state, self.default_Q)

    # Update Q-value for a particular state+action pair
    # (creating a new entry if necessary)
    def Q_writeOrUpdate(self, nAction, state, new_Q):
        # type: (int, BoardState, float) -> None
        buf = self.Q_tables[nAction]
        if state in buf:
            buf[state] = (1-self.alpha)*buf[state] + self.alpha*new_Q
        else:
            buf[state] = new_Q

    def train(self, history):
        raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - 
# train: (history:[(action:int, reward:float, state:BoardState)]) -> (none)
#     Use history (from the perspective of one player over the duration of one game) to update Q-table.
#     For each (state, action) pair, replace the entry (reward) in the Q-table with a one-step lookahead prediction.
#     i.e. 
#         - Apply the action to the state resulting in a new state s'
#         - from s' determine Q of the best action
#         - Q(state, action) <- reward(state, action) + gamma * Q(s', best action)

"""class MonteCarloAgent(TabularAgent):
    def train(self, history):
        if len(history) == 0:
            return
        prev_action, return_, _ = history[-1]
        for (action, reward, state) in reversed(history[:-1]):
            self.Q_writeOrUpdate(prev_action, state, return_)

            prev_action = action
            if reward is not None:
                return_ += reward"""
                
class TemporalDifferenceAgent(TabularAgent):
    def new_val(self, history, ndx):
        raise NotImplementedError()

    def train(self, history):
        if len(history) == 1:
            return
        for i in range(len(history)-2):
            (_, _, state), (action, _, _) = history[i:i+2]
            self.Q_writeOrUpdate(action, state, self.new_val(history, i))
            
        (_, _, state), (action, reward, _) = history[-2:]
        
        self.Q_writeOrUpdate(action, state, reward)

# - - - - - - - - - - - - - - - - - - - - - 
# One-step Q-table lookahead reward prediction

class QLearningAgent(TemporalDifferenceAgent):
    def new_val(self, history, ndx):
        (_, _, state), (action, reward, next_state)                      = history[ndx:ndx+2]
        next_action = self.max_action(next_state)
        return reward + self.gamma * self.Q_read(next_action, next_state)

"""class SarsaAgent(TemporalDifferenceAgent):
    def new_val(self, history, ndx):
        (_, _, state), (action, reward, next_state), (next_action, _, _) = history[ndx:ndx+3]
        return reward + self.gamma * self.Q_read(next_action, next_state)"""