import math
import random

class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        moves = game.available_moves()
        self.untried_moves = moves[:] if moves else [None]

    def ucb1(self, c=math.sqrt(2)):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def select_child(self):
        if not self.children:
            return None
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        if not self.untried_moves:
            return None
        move = self.untried_moves.pop()
        new_game = self.game.copy()
        # adiciona a opção de passar a vez
        new_game.make_move(move)
        child = MCTSNode(new_game, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result):
        self.visits += 1
        self.wins += result

def mcts(game, iterations=200):
    root = MCTSNode(game)

    for _ in range(iterations):
        node = root
        game_sim = game.copy()

        # Selection
        while node.untried_moves == [] and node.children:
            node = node.select_child()
            game_sim.make_move(node.move)

        # Expansion
        if node.untried_moves:
            node = node.expand()
            game_sim = node.game.copy()

        # Simulation
        while not game_sim.game_over():
            moves = game_sim.available_moves()
            if moves:
                move = random.choice(moves)
            else:
                move = None  # pass
            game_sim.make_move(move)

        # Backpropagation
        winner = game_sim.winner()
        if winner == 'X':
            result = 1
        elif winner == 'O':
            result = -1
        else:
            result = 0

        while node is not None:
            perspective = 1 if node.game.current == 'O' else -1
            node.update(perspective * result)
            node = node.parent

    if not root.children:
        return None
    best_child = max(root.children, key=lambda c: c.visits)
    return best_child.move
