import time
# import psutil as ps
# from scipy import stats

import argparse
import csv
import subprocess

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import multiprocessing
class Gatherer(multiprocessing.Process):
    def __init__(self, cpuQ, memoryQ):
        multiprocessing.Process.__init__(self)
        self.cpuQ = cpuQ
        self.memoryQ = memoryQ 

    def run(self):
        while(True):

            (output, err) = subprocess.Popen(['ps','u'], stdout=subprocess.PIPE).communicate()
            usage = output.decode('utf-8')
            usage = usage.split('\n') 
            c = 0
            m = 0
            for u in usage[1:]:
                a = np.array(u.split(' '))
                a = a[a != '']
                try:
                    c += float(a[2])
                    m += float(a[3])
                except:
                    pass
            if(c != 0):
                self.cpuQ.put(c)
            if(m != 0):
                self.memoryQ.put(m)

            # (output, err) = subprocess.Popen(['free'], stdout=subprocess.PIPE).communicate()
            # usage = output.decode('utf-8')
            # usage = usage.split('\n') 
            # a = np.array(usage[1].split(' '))
            # a = a[a != '']
            # try:
            #     used.append(a[2])
            #     free.append(a[3])
            #     shared.append(a[4])
            #     cache.append(a[5])
            #     available.append(a[6])
            # except:
            #     pass
            time.sleep(1)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('height',type=int,help='Height of the board')
parser.add_argument('width',type=int,help='Width of the board')
parser.add_argument('wincondition',type=int,help='How many in a row to win')
parser.add_argument('depth',type=int,help='Depth of the search, 0 = infinity')
parser.add_argument('iterations',type=int,help='How many simulations to run')
parser.add_argument('withpruning',type=int,help='True = 1, False = 0')
args = parser.parse_args()

# Settings
x = args.height
y = args.width
length = args.wincondition
iterations = args.iterations
withpruning = args.withpruning
# Gomoku Board
black = 1
blackTurn = True
white = -1
whiteTurn = False
blank = 0
if(args.depth <= 0):
    depth = float('inf')
else:
    depth = args.depth

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
        for i in range(1, board.length-1):
            for j in range(0, i+1):
                if([float(black)] * (i+1) == line[j:j+i+1]):
                    minMaxScore += i 
                if([float(white)] * (i+1) == line[j:j+i+1]):
                    minMaxScore -= i 
    return minMaxScore

def minimaxNoPruning(state, turn, depth):
    board = Board(state, x, y, length)

    if(turn == blackTurn):
        minMaxScore = float('-inf')
    elif(turn == whiteTurn):
        minMaxScore = float('inf')

    minMaxMove = None

    if(depth > 0 and not board.check_win() and not board.check_end()):
        moves = board.get_possibilities()
        depth -= 1
        for move in moves:
            player = black if turn else white
            state[move[0]][move[1]] = player
            score = minimaxNoPruning(np.copy(state), not turn, depth)[1]
            if(turn == blackTurn):
                if(minMaxScore <= score):
                    minMaxScore = score
                    minMaxMove = move
            elif(turn == whiteTurn):
                if(minMaxScore >= score):
                    minMaxScore = score
                    minMaxMove = move
            state[move[0]][move[1]] = blank
    else:
        if(board.check_win()):
            winner = int(board.check_who_win())
            if(winner == black):
                minMaxScore = float('inf')
            elif(winner == white):
                minMaxScore = float('-inf')
        else:
            minMaxScore = evaluate(board)
    
    return (minMaxMove, minMaxScore)

def minimax(state, turn, alpha, beta, depth):
    board = Board(state, x, y, length)

    if(turn == blackTurn):
        minMaxScore = float('-inf')
    elif(turn == whiteTurn):
        minMaxScore = float('inf')

    minMaxMove = None

    if(depth > 0 and not board.check_win() and not board.check_end()):
        moves = board.get_possibilities()
        depth -= 1
        for move in moves:
            player = black if turn else white
            state[move[0]][move[1]] = player
            score = minimax(np.copy(state), not turn, alpha, beta, depth)[1]
            if(turn == blackTurn):
                if(minMaxScore <= score):
                    minMaxScore = score
                    minMaxMove = move
                if(alpha <= score):
                    alpha = score
            elif(turn == whiteTurn):
                if(minMaxScore >= score):
                    minMaxScore = score
                    minMaxMove = move
                if(beta >= score):
                    beta = score
            state[move[0]][move[1]] = blank
            if(beta <= alpha):
                break
    else:
        if(board.check_win()):
            winner = int(board.check_who_win())
            if(winner == black):
                minMaxScore = float('inf')
            elif(winner == white):
                minMaxScore = float('-inf')
        else:
            minMaxScore = evaluate(board)
    
    return (minMaxMove, minMaxScore)

def parallelA(state, turn, depth):
    class workerThread(multiprocessing.Process):
        def __init__(self, state, move, q):
            multiprocessing.Process.__init__(self)
            self.state = state
            self.move = move
            self.q = q

        def run(self):
            state = self.state
            move = self.move
            q = self.q
            state[move[0]][move[1]] = black if turn else white 
            if(withpruning == 1):
                score = minimax(state, not turn, float('-inf'), float('inf'), depth)[1]
            if(withpruning == 0):
                score = minimaxNoPruning(state, not turn, depth)[1]
            minMaxScore = score
            minMaxMove = move 
            # Put move and score in the queue for parsing later
            q.put((minMaxScore, minMaxMove))

    board = Board(state, x, y, length)

    workers = []
    count = 0
    # Queue for keeping track of move scores
    q = multiprocessing.Queue()
    # Get possible moves (branches)
    moves = board.get_possibilities()
    for move in moves:
        # For each branch, create a process that traverses
        worker = workerThread(np.copy(state), move, q)
        count += 1
        worker.start()
        workers.append(worker)

    # Wait for all workers to be done before proceeding
    while len(workers) > 0:
        workers = [worker for worker in workers if worker.is_alive()]

    if(turn == blackTurn):
        minMaxScore = float('-inf')
    elif(turn == whiteTurn):
        minMaxScore = float('inf')
    
    minMaxMove = None
    
    # Go through the queue, determine which move has the highest score
    while(not q.empty()):
        item = q.get()
        score = item[0]
        move = item[1]
        if(turn == blackTurn):
            if(minMaxScore <= score):
                minMaxScore = score
                minMaxMove = move
        elif(turn == whiteTurn):
            if(minMaxScore >= score):
                minMaxScore = score
                minMaxMove = move

    return (minMaxMove, minMaxScore)

def gameMove(turn, player):
    # alpha = float('-inf')
    # beta = float('inf')

    state = np.copy(board.get_board())

    # if(withpruning == 1):
    #     data = minimax(state, turn, alpha, beta, depth)
    # if(withpruning == 0):
    #     data = minimaxNoPruning(state, turn, depth)
    data = parallelA(state, turn, depth)
    move = data[0]
    board.set_position(move[0], move[1], player)
    print(board.get_board())

cpuQ = multiprocessing.Queue()
memoryQ = multiprocessing.Queue()
gatherer = Gatherer(cpuQ, memoryQ)
gatherer.start()

for i in range(iterations):
    print('Iteration',i)
    state = np.zeros((x,y))
    board = Board(state,x,y,length)
    start_time = time.time()
    # gameRunning = True
    # while(gameRunning):
    #     if(gameRunning):
    gameMove(blackTurn, black)
        # if(board.check_win() or board.check_end()):
        #     gameRunning = False
        # if(gameRunning):
    gameMove(whiteTurn, white)
        # if(board.check_win() or board.check_end()):
        #     gameRunning = False
    end_time = time.time()
    times.append(end_time - start_time)

gatherer.terminate()

dfTimes = pd.DataFrame(times)
print('Average Time:',dfTimes.mean()[0])

cpu = []
memory = []

while(not cpuQ.empty()):
    cpu.append(cpuQ.get())

while(not memoryQ.empty()):
    memory.append(memoryQ.get())

if(len(cpu) > 0):
    cpuMean = pd.DataFrame(cpu).mean()[0]
    cpuHigh = pd.DataFrame(cpu).max()[0]
    cpuLow = pd.DataFrame(cpu).min()[0]
    cpuMedian = pd.DataFrame(cpu).median()[0]
else:
    cpuMean = 0
    cpuHigh = 0
    cpuLow = 0
    cpuMedian = 0

if(len(memory) > 0):
    memoryMean = pd.DataFrame(memory).mean()[0]
else:
    memoryMean = 0

with open('Parallel.csv', 'a') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([x,y,length,withpruning,depth] + times + [cpuMean,cpuHigh,cpuLow,cpuMedian,memoryMean])

# with open('SerialUsedMemory.csv', 'a') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow([x,y,length,withpruning] + used)
# with open('SerialCacheMemory.csv', 'a') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow([x,y,length,withpruning] + cache)
# with open('SerialShared.csv', 'a') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow([x,y,length,withpruning] + shared)