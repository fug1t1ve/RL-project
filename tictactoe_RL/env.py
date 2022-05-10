import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.pyplot import *
import numpy as np


from IPython.core.debugger import set_trace



class BoardState(object):
    winning_spots = np.array([
        [0, 1, 2], [3, 4, 5], [6, 7, 8], 
        [0, 3, 6], [1, 4, 7], [2, 5, 8], 
        [0, 4, 8], [2, 4, 6]             
        ])
    
    board_format = '\n'.join([
        ' {} | {} | {} ',
        '---+---+---',
        ' {} | {} | {} ',
        '---+---+---',
        ' {} | {} | {} ',
        ])

    def __init__(self, prev=None, action=None):
        if prev is not None:
            self.marks = prev.marks.copy()
            self.marks[action] = prev.active_player
            self.active_player = 'X' if prev.active_player == 'O' else 'O'
        else:
            self.active_player = 'X'
            self.marks = np.array(['_']*9)

    def __repr__(self):
        return ''.join(self.marks) + ',' + self.active_player

    def __str__(self):
        return BoardState.board_format.format(*self.marks)

    def __eq__(self, other):
        return isinstance(other, self.__class__) \
            and np.array_equal(self.marks, other.marks) \
            and self.active_player == other.active_player

    def __hash__(self):
        return hash(repr(self))

    @staticmethod
    def from_repr(s):
        out = BoardState()
        out.active_player = s[-1]
        out.marks = np.array(list(s[:-2]))
        return out

    def render(self):
        print(self.__str__())

    def step(self, action:int):      
        next_state = BoardState(self, action)
        reward = -1.0 if self.marks[action] != '_' \
            else +1.0 if next_state.check_win(self.active_player) \
            else  0.0

        done = next_state.is_full()  or  reward != 0.0
        
        return (next_state, reward, done)

    def check_win(self, player):
        slices = self.marks[BoardState.winning_spots]
        return (slices == player).all(axis=1).any()

    def is_full(self):
        return (self.marks != '_').all()
    
    def draw(self, agent):
        fig = figure(figsize=[3,3])
        ax = fig.add_subplot(111)

        def draw_cell(pos, mark, val):
            y, x = divmod(pos, 3)
            slices = self.marks[self.winning_spots]
            O = (slices == 'O').all(axis=1)
            X = (slices == 'X').all(axis=1)
            winningSliceIndices = np.append( O.nonzero(), X.nonzero() )
            winningSquares = np.unique( self.winning_spots[ winningSliceIndices ] )           
            
            if pos in winningSquares:
                ax.add_patch(patches.Rectangle((x,y), 1, 1, ec='none', fc='red'))
                            
            if mark == 'X':
                ax.plot([x+.2, x+.8], [y+.8, y+.2], 'k', lw=2.0)
                ax.plot([x+.2, x+.8], [y+.2, y+.8], 'k', lw=2.0)
            elif mark == 'O':
                ax.add_patch(patches.Circle((x+.5,y+.5), .35, ec='k', fc='none', lw=2.0))
            else:
                color = cm.viridis((val+1)/2.)
                ax.add_patch(patches.Rectangle((x,y), 1, 1, ec='none', fc=color))
                ax.text(x+.5 , y+.5, '%.2f'%val    , ha='center', va='center') 
                ax.text(x+.08, y+.12,  '%d'%(pos+1), ha='center', va='center') 

        for i in range(9):
            draw_cell(i, self.marks[i], agent.Q_read(i,self))

        ax.set_position([0,0,1,1])
        ax.set_axis_off()

        ax.set_xlim(0,3)
        ax.set_ylim(3,0)

        for x in range(1,3):
            ax.plot([x, x], [0,3], 'k', lw=2.0)
            ax.plot([0,3], [x, x], 'k', lw=2.0)
        show()