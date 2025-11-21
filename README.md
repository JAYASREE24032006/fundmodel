# 1.Developing AI Agent with PEAS Description

```
import random

class VacuumEnvironment:
    def __init__(self):
        self.locations = {'A': random.choice(['Clean', 'Dirty']),
                          'B': random.choice(['Clean', 'Dirty'])}
        self.agent_location = random.choice(['A', 'B'])
        self.performance = 0

    def sense(self):
        return self.agent_location, self.locations[self.agent_location]

    def execute(self, action):
        if action == "Left":
            self.agent_location = 'A'
            self.performance -= 1
        elif action == "Right":
            self.agent_location = 'B'
            self.performance -= 1
        elif action == "Suck":
            if self.locations[self.agent_location] == "Dirty":
                self.locations[self.agent_location] = "Clean"
                self.performance += 10
        elif action == "NoOp":
            if all(v == "Clean" for v in self.locations.values()):
                self.performance += 5

class ReflexAgent:
    def decide(self, location, status):
        if status == "Dirty":
            return "Suck"
        if location == "A":
            return "Right"
        return "Left"

env = VacuumEnvironment()
agent = ReflexAgent()

for _ in range(10):
    loc, stat = env.sense()
    action = agent.decide(loc, stat)
    env.execute(action)
    if all(v == "Clean" for v in env.locations.values()):
        env.execute("NoOp")
        break

print("Environment:", env.locations)
print("Location:", env.agent_location)
print("Performance:", env.performance)

```

# 2.Implement Depth First Search Function For a Graph

```
def dfs(graph,start,visited,path):
    path.append(start)
    visited[start]=True
    for neighbour in graph[start]:
        if visited[neighbour]==False:
            dfs(graph,neighbour,visited,path)
            visited[neighbour]=True
    return path
```

# 3.Implement Breadth First Search Traversal of a Graph as a Function

```
def bfs(graph,start,visited,path):
    queue = deque()
    path.append(start)
    queue.append(start)
    visited[start] = True
    while len(queue) != 0:
        tmpnode = queue.popleft()
        for neighbour in graph[tmpnode]:
            if visited[neighbour] == False:
                path.append(neighbour)
                queue.append(neighbour)
                visited[neighbour] = True
    return path
```

# 4.Implement A* search algorithm for a custom Graph

```
def a_star(graph, start, goal, heuristic):
    open_set = {start}
    closed_set = set()
    g = {start: 0}
    parent = {start: start}

    while open_set:
        n = min(open_set, key=lambda x: g[x] + heuristic(x))

        if n == goal:
            path = []
            while parent[n] != n:
                path.append(n)
                n = parent[n]
            path.append(start)
            return path[::-1]

        open_set.remove(n)
        closed_set.add(n)

        for (m, cost) in graph[n]:
            if m not in open_set and m not in closed_set:
                open_set.add(m)
                parent[m] = n
                g[m] = g[n] + cost
            else:
                if g[m] > g[n] + cost:
                    g[m] = g[n] + cost
                    parent[m] = n
                    if m in closed_set:
                        closed_set.remove(m)
                        open_set.add(m)

    return None
```

# 5.Implement Simple Hill Climbing Algorithm 

```
import random
import string

def generate_random_solution(answer):
    return [random.choice(string.printable) for _ in range(len(answer))]

def evaluate(solution, answer):
    target = list(answer)
    diff = 0
    for i in range(len(target)):
        diff += abs(ord(solution[i]) - ord(target[i]))
    return diff

def mutate_solution(solution):
    ind = random.randint(0, len(solution) - 1)
    solution[ind] = random.choice(string.printable)
    return solution

def SimpleHillClimbing():
    answer = "Artificial Intelligence"
    best = generate_random_solution(answer)
    best_score = evaluate(best, answer)
    while True:
        print("Score:", best_score, " Solution:", "".join(best))
        if best_score == 0:
            break
        new_solution = mutate_solution(list(best))
        score = evaluate(new_solution, answer)
        if score < best_score:
            best = new_solution
            best_score = score

SimpleHillClimbing()

```

# 6.Solve Cryptarithmetic Problem,a CSP(Constraint Satisfaction Problem) using Python

```
from itertools import permutations

def solve_cryptarithmetic():
    for perm in permutations(range(10), 8):
        S, E, N, D, M, O, R, Y = perm

        if S == 0 or M == 0:
            continue

        SEND = 1000 * S + 100 * E + 10 * N + D
        MORE = 1000 * M + 100 * O + 10 * R + E
        MONEY = 10000 * M + 1000 * O + 100 * N + 10 * E + Y

        if SEND + MORE == MONEY:
            return SEND, MORE, MONEY

    return None

solution = solve_cryptarithmetic()

if solution:
    SEND, MORE, MONEY = solution
    print(f'SEND = {SEND}')
    print(f'MORE = {MORE}')
    print(f'MONEY = {MONEY}')
else:
    print("No solution found.")
```

# 7.Implement Mini Max Search Algorithm

```
def max():
    maxv = -2
    px = None
    py = None
    result = is_end()
    if result == 'X':
        return (-1, 0, 0)
    elif result == 'O':
        return (1, 0, 0)
    elif result == '.':
        return (0, 0, 0)
    for i in range(0, 3):
        for j in range(0, 3):
            if current_state[i][j] == '.':
                current_state[i][j] = 'O'
                (m, min_i, min_j) = min()
                if m > maxv:
                    maxv = m
                    px = i
                    py = j
                current_state[i][j] = '.'
    return (maxv, px, py)
def min():
    minv = 2
    qx = None
    qy = None
    result = is_end()
    if result == 'X':
        return (-1, 0, 0)
    elif result == 'O':
        return (1, 0, 0)
    elif result == '.':
        return (0, 0, 0)
    for i in range(0, 3):
        for j in range(0, 3):
            if current_state[i][j] == '.':
                current_state[i][j] = 'X'
                (m, max_i, max_j) = max()
                if m < minv:
                    minv = m
                    qx = i
                    qy = j
                current_state[i][j] = '.'
    return (minv, qx, qy)
```

# 8.Implement Alpha-beta pruning of Minimax Search Algorithm for a Simple TIC-TAC-TOE game

```
board = [['.','.','.'],
         ['.','.','.'],
         ['.','.','.']]

def check():
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '.': return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] != '.': return board[0][i]
    if board[0][0] == board[1][1] == board[2][2] != '.': return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] != '.': return board[0][2]
    for i in range(3):
        for j in range(3):
            if board[i][j] == '.': return None
    return '.'

def ab(alpha, beta, is_max):
    r = check()
    if r == 'O': return (1, None, None)
    if r == 'X': return (-1, None, None)
    if r == '.': return (0, None, None)

    if is_max:
        best = -2; move=(None,None)
        for i in range(3):
            for j in range(3):
                if board[i][j]=='.':
                    board[i][j]='O'
                    val,_,_ = ab(alpha,beta,False)
                    board[i][j]='.'
                    if val>best: best,val; move=(i,j)
                    alpha = max(alpha,val)
                    if alpha>=beta: return (best,move[0],move[1])
        return (best,move[0],move[1])
    else:
        best = 2; move=(None,None)
        for i in range(3):
            for j in range(3):
                if board[i][j]=='.':
                    board[i][j]='X'
                    val,_,_ = ab(alpha,beta,True)
                    board[i][j]='.'
                    if val<best: best=val; move=(i,j)
                    beta = min(beta,val)
                    if alpha>=beta: return (best,move[0],move[1])
        return (best,move[0],move[1])

def show():
    for r in board: print(*r)
    print()

def play():
    turn='X'
    while True:
        show()
        r=check()
        if r:
            print("Result:",r)
            return
        if turn=='X':
            x=int(input("Row: "))
            y=int(input("Col: "))
            if board[x][y]=='.':
                board[x][y]='X'
                turn='O'
        else:
            _,x,y = ab(-2,2,True)
            board[x][y]='O'
            turn='X'

play()
```
