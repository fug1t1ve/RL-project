import os, sys, time ,random,pickle
from six.moves import zip_longest
from env import *
from agent import *

episodes = 10000
fname = 'tictac.txt'

def progressbar(callback, iters, refresh_rate=2.0):
    prev_clock = time.time()
    start_clock = prev_clock

    for i in range(iters):
        callback(i)
        curr_clock = time.time()
        if (curr_clock-prev_clock)*refresh_rate >= 1:
            sys.stdout.write('\r[ %s / %s ]' % (i, iters))
            sys.stdout.flush()
            prev_clock = curr_clock

    clearstr = ' '*len('[ %s / %s ]' % (iters, iters))
    sys.stdout.write('\r%s\r' % clearstr)
    sys.stdout.flush()

    return time.time() - start_clock

# Assumes zero-sum, two-player, sequential-turn game
def train_episode(agent, state=None):
    if state is None:
        # Start at a random previously encountered state
        keys = list( agent.Q_tables[random.randint(0,8)] )  # list of keys for Q-table dict
        if len(keys) > 0:
            state = random.choice(keys)
        else:
            state = BoardState()

    # Play out a game, recording each (action, reward, state) tuple.
    first_player = state.active_player
    history = [(None, None, state)]

    while True:
        action = agent.choose_action(state)
        state, reward, done = state.step(action)
        history.append((action, reward, state))
        if done:
            break

    # Split history into a separate history for each player.
    #    history stores things like [(None, None, s1), (p1a1, p1r1, s2), (p2a1, p2r1, s3), (p1a2, p1r2, s4), ...]
    #    player_history transforms that to [(None, None, s1), (p1a1, p1r1-p2r1, s3), (p1a2, p1r2-p2r2, s5), ...]
    #    You subtract the reward given to the other player because of the assumption of it being a zero-sum game.
    #    (Think: relative reward)
    def player_history(history):
        # e.g.  grouped('ABCDEFG', 3, 'x') --> 'ABC' 'DEF' 'Gxx'
        def grouped(iterable, n, fillvalue=None):
            "Collect data into fixed-length chunks or blocks"
            # https://docs.python.org/2/library/itertools.html#recipes
            args = [iter(iterable)] * n
            return zip_longest(fillvalue=fillvalue, *args)

        out = [(None, None, history[0][2])]
        for (action, reward, state), (_, other_reward, other_state) \
                                in grouped(history[1:], 2, (None,)*3):
            if other_reward is None:
                out.append((action, reward, state))
            else:
                out.append((action, reward-other_reward, other_state))
        return out
    
    first_history = player_history(history)
    second_history = player_history(history[1:])
    
    # Update Q-tables
    agent.train(first_history)
    agent.train(second_history)

    
if os.path.isfile(fname):
    print('Loading agent from %s...' % fname)
    agent = pickle.load(open(fname, 'rb'))
else:
    agent = QLearningAgent(num_actions=9, epsilon=0.8, default_Q=2)

#init_state = BoardState() # Always start from actual inital state
init_state = None # Random restarts

print('Training for %d episodes...' % episodes)
progressbar(lambda x: train_episode(agent, init_state), episodes)

print('Saving agent to %s...' % fname)
pickle.dump(agent, open(fname, 'wb'))