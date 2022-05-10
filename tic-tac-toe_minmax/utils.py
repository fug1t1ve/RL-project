from treelib import Node, Tree
from env import *
import random 
tree = Tree()
tree.create_node("root", "root")
tree.create_node("", "l1", parent='root')
tree.create_node("", "r1", parent='root')
tree.create_node("3", "l1-1", parent='l1')
tree.create_node("5", "l1-2", parent='l1')
tree.create_node("2", "r1-1", parent='r1')
tree.create_node("9", "r1-2", parent='r1')
tree
tree.show()

def minmax(tree, current_id, is_max):
    if tree.depth(current_id) == tree.depth():             
        return int(tree[current_id].tag)                   
    children_of_current_id = tree.children(current_id)     
    scores = [minmax(tree, child.identifier, not is_max) for child in children_of_current_id]   
    if is_max:                                             
        return max(scores)
    else:
        return min(scores)

minmax(tree, 'root', True)

import copy

def remove_value_list(l, val):
    return [el for el in l if el != val]

flip_player = {1: -1, -1: 1}

possible_options = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

def add_options_to_node(tree, node, tt_data, player, remaining_options):
    for option in remaining_options:
        local_tt_data = copy.deepcopy(tt_data)           
        local_tt_data.make_move(player, option, False)
        if node.identifier != 'root':
            new_identifier = node.identifier + option
        else:
            new_identifier = option
        tree.create_node(option, new_identifier, node.identifier, data = local_tt_data)
        if len(remaining_options) > 1 and not local_tt_data.is_endstate():  
            add_options_to_node(tree, tree[new_identifier], local_tt_data, 
                                flip_player[player], remove_value_list(remaining_options, option))
    return None

TicToe_state = Tictoe(3)
TicToe_3x3 = Tree()
TicToe_3x3.create_node("root", "root")
add_options_to_node(TicToe_3x3, TicToe_3x3["root"], 
                    TicToe_state, 1, possible_options)

def minmax_tt(tree, current_id, is_max):
    current_node = tree[current_id]                     
    if current_node.data.is_endstate():                 
        return current_node.data.get_value()            
    children_of_current_id = tree.children(current_id)  
    scores = [minmax_tt(tree, child.identifier, not is_max) for child in children_of_current_id]   
    if is_max:                                         
        return max(scores)
    else:
        return min(scores)
    
def determine_move(tree, current_id, is_max):
    potential_moves = tree.children(current_id)
    moves = [child.identifier[-1] for child in potential_moves]
    raw_scores = np.array([minmax_tt(tree, child.identifier, not is_max) for child in potential_moves])
    if is_max:
        return moves[random.choice(np.where(raw_scores == max(raw_scores))[0])]
    else:
        return moves[random.choice(np.where(raw_scores == min(raw_scores))[0])]

tictactoe = Tictoe(3)

print('''Welcome to TicTacToe. 

You can make a move by selecting one of the following letters:''')
print(np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']).reshape(3,3))
print('''You start, the computer will take the next move

Initial board:''')

move_history = ''
while not tictactoe.is_endstate():
    player_move = input('Your move!: ')
    tictactoe.make_move(1, player_move)
    print(tictactoe.get_board())
    move_history += player_move
    if tictactoe.is_endstate():
        print('You won!...wait you won?????')
    
    print('Computer is thinking')
    computer_move = determine_move(TicToe_3x3, move_history, False)
    tictactoe.make_move(-1, computer_move)
    print(tictactoe.get_board())
    move_history += computer_move
    if tictactoe.is_endstate():
        print('Computer won!')
        
    if len(move_history) >= 8 and not tictactoe.is_endstate():
        print('Draw...')
        break