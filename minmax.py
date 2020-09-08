import time
# import psutil as ps
# from scipy import stats

import argparse
import csv

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import multiprocessing

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('height',type=int,help='Height of the board')
parser.add_argument('width',type=int,help='Width of the board')
parser.add_argument('wincondition',type=int,help='How many in a row to win')
parser.add_argument('iterations',type=int,help='How many simulations to run')
parser.add_argument('implementation',type=int,help='Serial = 1, Parallel = 2')
args = parser.parse_args()

# Settings
x = args.height
y = args.width
length = args.wincondition
method = args.implementation
iterations = args.iterations
if(method == 1):
    methodName = 'Serial'
elif(method == 2):
    methodName = 'Parallel'

# Gomoku Board
black = 1
blackTurn = True
white = -1
whiteTurn = False
blank = 0
depth = 2

minMaxMove = None
minMaxScore = None

times = []

class Board:

    def __init__(self, board, x, y, length):
        self.board = board
        self.length = length 
        self.height = y
        self.width = x 
    
    def get_board(self):
        return self.board

    def get_position(self, x, y):
        return self.board[x][y]

    def set_position(self, x, y, z):
        self.board[x][y] = z

    def get_column(self, y, x, length):
        line = np.empty(length)
        for i in range(length):
            line[i] = self.board[y+i,x]
        return line, [(y+i,x) for i in range(length)]

    def get_row(self, y, x, length):
        line = np.empty(length)
        for i in range(length):
            line[i] = self.board[y,x+i]
        return line, [(y,x+i) for i in range(length)]

    def get_diagonal_upleft_to_lowright(self, y, x, length):
        line = np.empty(length)
        for i in range(length):
            line[i] = self.board[y+i,x+i]
        return line, [(y+i,x+i) for i in range(length)]

    def get_diagonal_lowleft_to_upright(self, y, x, length):
        line = np.empty(length)
        if y < length - 1:
            raise IndexError
        for i in range(length):
            line[i] = self.board[y-i,x+i]
        return line, [(y-i,x+i) for i in range(length)]

    def check_win(self):
        for i in range(self.height):
            for j in range(self.width):
                for getter_function in (self.get_row, self.get_column, self.get_diagonal_lowleft_to_upright, self.get_diagonal_upleft_to_lowright):
                    try:
                        line, positions = getter_function(i,j,self.length)
                    except IndexError:
                        continue
                    if abs(line.sum()) == self.length:
                        return True
        return False

    def check_who_win(self):
        for i in range(self.height):
            for j in range(self.width):
                for getter_function in (self.get_row, self.get_column, self.get_diagonal_lowleft_to_upright, self.get_diagonal_upleft_to_lowright):
                    try:
                        line, positions = getter_function(i,j,self.length)
                    except IndexError:
                        continue
                    if abs(line.sum()) == self.length:
                        return line[0]

    def check_end(self):
        possibilities = np.where(self.board == 0)
        px = possibilities[0]
        py = possibilities[1]
        possibilities = np.concatenate((px,py)).reshape(-1,2,order='F')
        return len(possibilities) == 0

    def check_consecutive(self):
        sums = []
        consecutives = []
        for i in range(self.height):
            for j in range(self.width):
                for getter_function in (self.get_row, self.get_column, self.get_diagonal_lowleft_to_upright, self.get_diagonal_upleft_to_lowright):
                    try:
                        line, positions = getter_function(i,j,self.length)
                    except IndexError:
                        continue
                    if abs(line.sum()) > 0:
                        sums.append(line.sum())
                        consecutives.append([self.board[p[0]][p[1]] for p in positions])
        # return (np.array(sums), consecutives)
        return consecutives
    
    def get_possibilities(self):
        possibilities = np.where(self.board == 0)
        px = possibilities[0]
        py = possibilities[1]
        return np.concatenate((px,py)).reshape(-1,2,order='F')

def evaluate(board):
    consecutives = board.check_consecutive()
    minMaxScore = 0
    for line in consecutives:
        # if([black] in line):
        #     minMaxScore += 0.5
        # if([white] in line):
        #     minMaxScore -= 0.5 
        for i in range(1, board.length-1):
            for j in range(0, i+1):
                if([float(black)] * (i+1) == line[j:j+i+1]):
                    minMaxScore += i 
                if([float(white)] * (i+1) == line[j:j+i+1]):
                    minMaxScore -= i 
    return minMaxScore

def minimax(state, move, turn, depth):
    player = black if turn else white
    depth -= 1
    state[move[0]][move[1]] = player
    board = Board(state, x, y, length)

    if(depth > 0 and not board.check_win() and not board.check_end()):
        moves = board.get_possibilities()
        scores = []
        for move in moves:
            scores.append(minimax(np.copy(state), move, not turn, depth))
        if(turn == blackTurn):
            minMaxScore = min(scores)
        elif(turn == whiteTurn):
            minMaxScore = max(scores)
    else:
        if(board.check_win()):
            winner = int(board.check_who_win())
            if(winner == black):
                minMaxScore = float('inf')
            elif(winner == white):
                minMaxScore = float('-inf')
        else:
            minMaxScore = evaluate(board)
    
    return minMaxScore

def serialMinimax(board, turn, depth):

    # Start Algorithm Section

    if(turn == blackTurn):
        minMaxScore = float('-inf')
    elif(turn == whiteTurn):
        minMaxScore = float('inf')

    minMaxMove = None

    # Get possible moves
    count = 0
    moves = board.get_possibilities()
    for move in moves:
        # Evaluate the current board state
        score = minimax(np.copy(board.get_board()), move, turn, depth)
        count += 1
        if(turn == blackTurn):
            if(minMaxScore <= score):
                minMaxScore = score
                minMaxMove = move
        elif(turn == whiteTurn):
            if(minMaxScore >= score):
                minMaxScore = score
                minMaxMove = move

    # End Algorithm Section
    # memory = ps.virtual_memory().percent
    data = [minMaxMove]
    return data

def parallelMinimax(board, turn, depth):
    class workerThread(multiprocessing.Process):
        def __init__(self, threadNum, turn, move, q):
            multiprocessing.Process.__init__(self)
            self.threadNum = threadNum
            self.turn = turn 
            self.move = move
            self.q = q

        def run(self):
            global minMaxMove
            global minMaxScore
            turn = self.turn
            move = self.move
            score = minimax(np.copy(board.get_board()), move, turn, depth)
            if(turn == blackTurn):
                if(minMaxScore < score):
                    minMaxScore = score
                    minMaxMove = move
            elif(turn == whiteTurn):
                if(minMaxScore > score):
                    minMaxScore = score
                    minMaxMove = move
            while(not q.empty()):
                item = q.get()
                score = item[0]
                move = item[1]
                if(turn == blackTurn):
                    if(minMaxScore < score):
                        minMaxScore = score
                        minMaxMove = move
                elif(turn == whiteTurn):
                    if(minMaxScore > score):
                        minMaxScore = score
                        minMaxMove = move
            q.put((minMaxScore, minMaxMove))

    # Start Algorithm Section
    global minMaxScore
    global minMaxMove

    if(turn == blackTurn):
        minMaxScore = float('-inf')
    elif(turn == whiteTurn):
        minMaxScore = float('inf')


    # Get possible moves
    moves = board.get_possibilities()
    workers = []
    count = 0
    q = multiprocessing.Queue()
    for move in moves:
        worker = workerThread(count, turn, move, q)
        count += 1
        worker.start()
        workers.append(worker)

    while len(workers) > 0:
        workers = [worker for worker in workers if worker.is_alive()]
    
    item = q.get()
    minMaxScore = item[0]
    minMaxMove = item[1]

    # End Algorithm Section
    # memory = ps.virtual_memory().percent
    data = [minMaxMove]
    return data

def blackToMove(method):
    if(method == 1):
        data = serialMinimax(board, blackTurn, depth)
    elif(method == 2):
        data = parallelMinimax(board, blackTurn, depth)
    move = data[0]
    board.set_position(move[0], move[1], black)
    print(board.get_board())


def whiteToMove(method):
    if(method == 1):
        data = serialMinimax(board, whiteTurn, depth)
    elif(method == 2):
        data = parallelMinimax(board, whiteTurn, depth)
    move = data[0]
    board.set_position(move[0], move[1], white)
    print(board.get_board())


print(methodName)

for i in range(iterations):
    print('Iteration',i)
    state = np.zeros((x,y))
    board = Board(state,x,y,length)
    start_time = time.time()
    gameRunning = True
    while(gameRunning):
        if(gameRunning):
            blackToMove(method)
        if(board.check_win() or board.check_end()):
            gameRunning = False
        if(gameRunning):
            whiteToMove(method)
        if(board.check_win() or board.check_end()):
            gameRunning = False
    end_time = time.time()
    times.append(end_time - start_time)

dfTimes = pd.DataFrame(times)
print('Average Time:',dfTimes.mean())

filename = methodName + ' (' + str(','.join([str(x),str(y),str(length),str(iterations)])) + ') '

with open(filename+'.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    for t in times:
        writer.writerow([t])

