from board_game import BoardGame

class Othello(BoardGame):
    def __init__(self):
        super().__init__(8, 8)
        # posição inicial
        mid1 = self.rows // 2 - 1  # 3
        mid2 = self.rows // 2      # 4
        self.board[mid1][mid1] = 'O'
        self.board[mid2][mid2] = 'O'
        self.board[mid1][mid2] = 'X'
        self.board[mid2][mid1] = 'X'
        self.current = 'X'
        self.dirs = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

    def on_board(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def opponent(self, player):
        return 'O' if player == 'X' else 'X'

    def legal_moves_for(self, player):
        moves = []
        opp = self.opponent(player)
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] != ' ':
                    continue
                # checar direções
                valid = False
                for dr, dc in self.dirs:
                    rr, cc = r + dr, c + dc
                    found_opp = False
                    while self.on_board(rr, cc) and self.board[rr][cc] == opp:
                        found_opp = True
                        rr += dr
                        cc += dc
                    if found_opp and self.on_board(rr, cc) and self.board[rr][cc] == player:
                        valid = True
                        break
                if valid:
                    moves.append((r, c))
        return moves

    def available_moves(self):
        return self.legal_moves_for(self.current)

    def make_move(self, move):
        player = self.current
        opp = self.opponent(player)

        # movimento de passar a vez
        if move is None:
            if self.legal_moves_for(player):
                return False
            self.current = opp
            return True

        r, c = move
        if not self.on_board(r, c) or self.board[r][c] != ' ':
            return False

        flips = []  # caminhos possiveis para flipar peças
        for dr, dc in self.dirs:
            rr, cc = r + dr, c + dc
            path = []
            while self.on_board(rr, cc) and self.board[rr][cc] == opp:
                path.append((rr, cc))
                rr += dr
                cc += dc
            if path and self.on_board(rr, cc) and self.board[rr][cc] == player:
                flips.extend(path)

        if not flips:
            return False  # movimento invalido se não flipa

        # flipagem
        self.board[r][c] = player
        for (fr, fc) in flips:
            self.board[fr][fc] = player

        self.current = opp
        return True

    def has_any_moves(self, player):
        return len(self.legal_moves_for(player)) > 0

    def full(self):
        return all(cell != ' ' for row in self.board for cell in row)

    def game_over(self):
        return not (self.has_any_moves('X') or self.has_any_moves('O')) or self.full()

    def score(self):
        x = sum(1 for row in self.board for cell in row if cell == 'X')
        o = sum(1 for row in self.board for cell in row if cell == 'O')
        return x, o

    def winner(self):
        if not self.game_over():
            return None
        x, o = self.score()
        if x > o:
            return 'X'
        elif o > x:
            return 'O'
        else:
            return None

    def copy(self):
        new = self.__class__()
        new.board = [row.copy() for row in self.board]
        new.current = self.current
        return new
