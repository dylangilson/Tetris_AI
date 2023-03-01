"""
* Dylan Gilson
* dylan.gilson@outlook.com
* March 1, 2023
"""

import cv2
import itertools
import numpy as np
from PIL import Image
import random


class Tetris:
    # constants
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    MAP_EMPTY = 8
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 700
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    NEXT_PIECE_BOARD_SIZE = 6
    BLOCK_SIZE = 30
    SCORE_OFFSET = 50
    BOARD_OFFSET = 100
    NEXT_PIECE_Y_OFFSET = 467
    NEXT_PIECE_X_OFFSET = 280
    NEXT_PIECE_BOARD_Y_OFFSET = 310
    NEXT_PIECE_BOARD_X_OFFSET = 60

    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        }
    }

    COLOURS = {
        0: (255, 0, 0),  # RED
        1: (0, 255, 0),  # GREEN
        2: (0, 0, 255),  # BLUE
        3: (255, 255, 0),  # YELLOW
        4: (255, 0, 255),  # PURPLE
        5: (0, 255, 255),  # CYAN
        6: (255, 165, 0),  # ORANGE
        7: (255, 255, 255),  # WHITE
        8: (0, 0, 0),  # BLACK
        9: (128, 128, 128)  # GREY
    }

    def __init__(self):
        self.game_over = False
        self.current_position = [3, 0]
        self.current_rotation = 0
        self.board = []
        self.board_colours = []
        self.next_piece_board = []
        self.next_piece_board_colours = []
        self.bag = []
        self.current_piece = None
        self.next_piece = None
        self.score = 0

        self.reset()

    def reset(self):
        self.board = [[self.MAP_EMPTY] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.board_colours = [[self.MAP_EMPTY] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.next_piece_board = [[self.MAP_EMPTY] * self.NEXT_PIECE_BOARD_SIZE
                                 for _ in range(self.NEXT_PIECE_BOARD_SIZE)]
        self.next_piece_board_colours = [[self.MAP_EMPTY] * self.NEXT_PIECE_BOARD_SIZE
                                         for _ in range(self.NEXT_PIECE_BOARD_SIZE)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self.new_round(piece_fall=False)
        self.score = 0

        return self.get_board_properties(self.board)

    def get_rotated_piece(self, piece, rotation):
        return self.TETROMINOS[piece][rotation]

    def update_board(self):
        piece = self.get_rotated_piece(self.current_piece, self.current_rotation)
        piece = [np.add(x, self.current_position) for x in piece]
        board = [x[:] for x in self.board]

        for x, y in piece:
            board[y][x] = self.MAP_PLAYER
            self.board_colours[y][x] = self.current_piece

    def update_board_colours(self):
        for i in range(self.BOARD_WIDTH):
            for j in range(self.BOARD_HEIGHT):
                if self.board[j][i] == self.MAP_EMPTY:
                    self.board_colours[j][i] = self.MAP_EMPTY

    def get_next_piece(self):
        piece = self.get_rotated_piece(self.next_piece, 0)
        piece = [np.add(x, [1, 2]) for x in piece]
        board = [x[:] for x in self.next_piece_board]

        for x, y in piece:
            board[y][x] = self.MAP_BLOCK
            self.next_piece_board_colours[y][x] = self.next_piece

    def update_next_piece_board_colours(self):
        for i in range(self.NEXT_PIECE_BOARD_SIZE):
            for j in range(self.NEXT_PIECE_BOARD_SIZE):
                self.next_piece_board_colours[j][i] = self.MAP_EMPTY

    def get_game_score(self):
        return self.score

    def new_round(self, piece_fall=False):
        score = 0
        
        if piece_fall:
            # update board and calculate score
            piece = self.get_rotated_piece(self.current_piece, self.current_rotation)
            self.board = self.add_piece_to_board(piece, self.current_position)
            lines_cleared, self.board = self.clear_rows(self.board)
            self.score += 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH

        # generate new bag
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_position = [3, 0]
        self.current_rotation = 0

        if not self.is_valid_position(self.get_rotated_piece(self.current_piece, self.current_rotation),
                                      self.current_position):
            self.game_over = True
            print('Final Score: ' + str(self.score))

        return score

    # return True if the position is invalid
    def is_valid_position(self, piece, pos):
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            
            if x < 0 or x >= Tetris.BOARD_WIDTH or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == self.MAP_BLOCK:
                return False
            
        return True

    def rotate(self, angle):
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def add_piece_to_board(self, piece, position):
        board = [x[:] for x in self.board]
        
        for x, y in piece:
            board[y + position[1]][x + position[0]] = self.MAP_BLOCK
            
        return board

    def clear_rows(self, board):
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]

        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]

            for _ in lines_to_clear:
                board.insert(0, [self.MAP_EMPTY for _ in range(Tetris.BOARD_WIDTH)])

        return len(lines_to_clear), board

    # get number of holes in board (empty square with at least one block above it)
    def number_of_holes(self, board):
        holes = 0

        for column in zip(*board):
            tail = itertools.dropwhile(lambda x: x != self.MAP_BLOCK, column)
            holes += len([x for x in tail if x == Tetris.MAP_EMPTY])

        return holes

    # get sum of differences of heights between pair of columns
    def bumpiness(self, board):
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for column in zip(*board):
            tail = itertools.dropwhile(lambda x: x != self.MAP_BLOCK, column)
            n = Tetris.BOARD_HEIGHT - len([x for x in tail])
            min_ys.append(n)

        for (y0, y1) in self.window(min_ys):
            bumpiness = abs(y0 - y1)
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    # get sum and max height of board
    def height(self, board):
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for column in zip(*board):
            tail = itertools.dropwhile(lambda x: x != self.MAP_BLOCK, column)
            height = len([x for x in tail])

            sum_height += height
            max_height = max(height, max_height)
            min_height = min(height, min_height)

        return sum_height, max_height, min_height

    def get_board_properties(self, board):
        lines, board = self.clear_rows(board)
        holes = self.number_of_holes(board)
        total_bumpiness, max_bumpiness = self.bumpiness(board)
        sum_height, max_height, min_height = self.height(board)
        return [lines, holes, total_bumpiness, sum_height]

    # get all possible next states
    def get_next_states(self):
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # for all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                position = [x, 0]

                # drop piece
                while self.is_valid_position(piece, position):
                    position[1] += 1
                position[1] -= 1

                # valid move
                if position[1] >= 0:
                    board = self.add_piece_to_board(piece, position)
                    states[(x, rotation)] = self.get_board_properties(board)

        return states

    def get_state_size(self):
        return 4

    def move(self, shift_m, shift_r):
        position = self.current_position.copy()
        position[0] += shift_m[0]
        position[1] += shift_m[1]
        rotation = self.current_rotation
        rotation = (rotation + shift_r + 360) % 360
        piece = self.get_rotated_piece(self.current_piece, rotation)

        if self.is_valid_position(piece, position):
            self.current_position = position
            self.current_rotation = rotation
            return True

        return False

    # return True if a fall move is possible
    def fall(self) -> bool:
        if not self.move([0, 1], 0):
            self.new_round(piece_fall=True)

            if self.game_over:
                self.score -= 2

        return self.game_over

    def hard_drop(self, position, rotation, render=False):
        self.current_position = position
        self.current_rotation = rotation

        # drop piece
        piece = self.get_rotated_piece(self.current_piece, self.current_rotation)
        while self.is_valid_position(piece, self.current_position):
            if render:
                self.render(wait_key=True)
            self.current_position[1] += 1
        self.current_position[1] -= 1

        piece = [np.add(x, self.current_position) for x in piece]
        board = [x[:] for x in self.board]

        for x, y in piece:
            board[y][x] = self.MAP_BLOCK
            self.board_colours[y][x] = self.current_piece

        # start new round
        score = self.new_round(piece_fall=True)
        if self.game_over:
            score -= 2

        if render:
            self.render(wait_key=True)

        return score, self.game_over

    def render_grid(self, img, border_colour):
        x = 0
        y = 0

        while x <= img.shape[1]:
            colour = self.COLOURS[7]
            thickness = 2

            if x % img.shape[1] == 0:
                colour = border_colour
                colour = colour[::-1]
                thickness = 4

            cv2.line(img, (x, 0), (x, img.shape[0]), color=colour, lineType=cv2.LINE_AA, thickness=thickness)
            x += self.BLOCK_SIZE

        while y <= img.shape[0]:
            colour = self.COLOURS[7]
            thickness = 2

            if y % img.shape[0] == 0:
                colour = border_colour
                colour = colour[::-1]
                thickness = 4

            cv2.line(img, (0, y), (img.shape[1], y), color=colour, lineType=cv2.LINE_AA, thickness=thickness)
            y += self.BLOCK_SIZE

    def prepare_image(self, img, board_height, board_width):
        img = np.array(img).reshape((board_height, board_width, 3)).astype(np.uint8)
        img = img[..., ::-1]  # convert RGB to BGR
        img = Image.fromarray(img, 'RGB')
        img = img.resize((board_width * self.BLOCK_SIZE, board_height * self.BLOCK_SIZE))
        img = np.array(img)

        return img

    def render(self, wait_key=False):
        self.update_board_colours()
        self.update_board()
        board = [Tetris.COLOURS[p] for row in self.board_colours for p in row]
        board = self.prepare_image(board, Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH)
        self.render_grid(board, (255, 0, 0))
        board = cv2.copyMakeBorder(board, self.BOARD_OFFSET, self.BOARD_OFFSET, self.BOARD_OFFSET, 0,
                                   cv2.BORDER_CONSTANT, value=self.COLOURS[8])

        self.update_next_piece_board_colours()
        self.get_next_piece()
        next_piece_board = [Tetris.COLOURS[p] for row in self.next_piece_board_colours for p in row]
        next_piece_board = self.prepare_image(next_piece_board, self.NEXT_PIECE_BOARD_SIZE, self.NEXT_PIECE_BOARD_SIZE)
        self.render_grid(next_piece_board, (0, 0, 255))
        next_piece_board = cv2.copyMakeBorder(next_piece_board, self.NEXT_PIECE_BOARD_Y_OFFSET,
                                              self.NEXT_PIECE_BOARD_Y_OFFSET, self.NEXT_PIECE_BOARD_X_OFFSET,
                                              self.NEXT_PIECE_BOARD_X_OFFSET, cv2.BORDER_CONSTANT,
                                              value=self.COLOURS[8])

        output = np.concatenate((board, next_piece_board), axis=1)

        cv2.putText(output, 'Score: ' + str(self.score), (self.SCORE_OFFSET, self.SCORE_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOURS[7], 2)
        cv2.putText(output, 'Next Piece', (self.NEXT_PIECE_Y_OFFSET, self.NEXT_PIECE_X_OFFSET),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.COLOURS[7], 2)
        cv2.imshow('Tetris', np.array(output))

        if wait_key:
            cv2.waitKey(1)  # allows for rendering during training

    # return a sliding window of width n over data
    def window(self, seq, n=2):
        iterable = iter(seq)
        result = tuple(itertools.islice(iterable, n))

        if len(result) == n:
            yield result

        for element in iterable:
            result = result[1:] + (element, )
            yield result
