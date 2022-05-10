from Game import *
from Dino import *
from model import *
from train import *
from init  import*
def playGame(observe=False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_sate(dino,game)    
    model = buildmodel()
    try:
        trainNetwork(model,game_state,observe=observe)
    except StopIteration:
        game.end()
init_cache()
playGame(observe=False)