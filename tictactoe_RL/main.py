import random, pickle
from env import *

def play_vs_human(agent, state=None):
    if state is None:
        state = BoardState()    
    state.draw(agent)
    
    # Flip a coin for who goes first
    compToMove = random.random() > 0.5
    
    while True:
        if compToMove:
            state, reward, done = state.step(agent.choose_action(state))
        else:
            move = int(input('Choose your move [1-9]: ')) - 1
            state, reward, done = state.step(move)
            
        state.draw(agent)
        
        if done:
            if compToMove:
                s = 'Tie!' if reward == 0  else   'I win!' if reward > 0  else   'I lose!'
            else:
                s = 'Tie!' if reward == 0  else 'You win!' if reward > 0  else 'You lose!'
                
            input(s + "\nPress Enter to play again...")
            return
            
        compToMove = not compToMove

fname = 'tictac.txt'

agent = pickle.load(open(fname, 'rb'))
agent.epsilon = 0.99
while True:
    play_vs_human(agent)