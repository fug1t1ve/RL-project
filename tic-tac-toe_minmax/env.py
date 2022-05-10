import numpy as np

class Tictoe:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros(size*size)
        self.letters_to_move = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'][:size*size]
    def get_board(self):
        return self.board.reshape([self.size, self.size])
    def make_move(self, who, where, verbose=False):
        self.board[self.letters_to_move.index(where)] = who
    def get_sums_of_board(self):
        local_board = self.get_board()
        return np.concatenate([local_board.sum(axis=0),             # columns
                               local_board.sum(axis=1),             # rows
                               np.trace(local_board),               # diagonal
                               np.trace(np.fliplr(local_board))], axis=None)   # other diagonal
    def is_endstate(self):
        someone_won = len(np.intersect1d((self.size, -self.size), self.get_sums_of_board())) > 0
        draw = np.count_nonzero(self.board) == self.size * self.size
        return someone_won or draw
    def get_value(self):
        sums = self.get_sums_of_board()
        if self.size in sums:
            return 10 - np.count_nonzero(self.board)   # Punish longer games
        elif -self.size in sums:
            return -10 + np.count_nonzero(self.board)
        else:
            return 0

