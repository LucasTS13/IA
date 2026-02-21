import time
import argparse
from colorama import Fore, Style, init
init(autoreset=True)

from board_othello import Othello
from helper_functions import colorize, clear_screen
from mcts import mcts
from minimax import minimax_with_hef

def evaluate_othello(board, player):
    opponent = 'O' if player == 'X' else 'X'

    # contagem de peças
    player_count = sum(1 for row in board for cell in row if cell == player)
    opp_count = sum(1 for row in board for cell in row if cell == opponent)
    piece_diff = player_count - opp_count

    # cantos
    corners = [(0,0),(0,7),(7,0),(7,7)]
    corner_score = 0
    for r,c in corners:
        if board[r][c] == player:
            corner_score += 25
        elif board[r][c] == opponent:
            corner_score -= 25

    # bordas
    edges = [(i,0) for i in range(8)] + [(i,7) for i in range(8)] + [(0,i) for i in range(8)] + [(7,i) for i in range(8)]
    edge_score = 0
    for r,c in edges:
        if board[r][c] == player:
            edge_score += 2
        elif board[r][c] == opponent:
            edge_score -= 2

    # mobilidade
    temp = Othello()
    temp.board = [row.copy() for row in board]
    temp.current = player
    player_moves = len(temp.available_moves())
    temp.current = opponent
    opp_moves = len(temp.available_moves())
    if player_moves + opp_moves > 0:
        mobility = 100 * (player_moves - opp_moves) / (player_moves + opp_moves)
    else:
        mobility = 0

    # pontuação total
    value = (10 * piece_diff) + corner_score + edge_score + (5 * mobility)
    return value


#  Melhor jogada para o Minimax
def best_move_minimax_othello(game, depth=4):
    player = game.current
    best_score = float('-inf')
    best_action = None

    for move in game.available_moves() or [None]:
        new_game = game.copy()
        new_game.make_move(move)
        val = minimax_with_hef(new_game, depth-1, maximizing=False, player=player, evaluate_fn=evaluate_othello)
        if val > best_score:
            best_score = val
            best_action = move
    return best_action


def print_board_othello(board):
    clear_screen()
    print(Fore.CYAN + "     COLUNAS")
    print("    " + " ".join(str(c+1) for c in range(8)))
    for r in range(8):
        row_display = " ".join(colorize(board[r][c]) if board[r][c] != ' ' else '.' for c in range(8))
        print(f"L{r+1:<2}  {row_display}")
    print(Style.DIM + "\n'.' indica casa vazia.\n")


def play(algorithm: str):
    game = Othello()
    player = input("Escolha seu lado (X ou O): ").strip().upper()
    assert player in ['X', 'O']
    opp = 'O' if player == 'X' else 'X'

    while not game.game_over():
        print_board_othello(game.board)
        x, o = game.score()
        print(f"{Fore.RED}X: {x}  {Fore.YELLOW}O: {o}  {Style.RESET_ALL}")
        print(f"Vez do jogador: {Fore.GREEN}{game.current}{Style.RESET_ALL}")

        if game.current == player:
            moves = game.available_moves()
            if not moves:
                print("Você não tem jogadas válidas — passando turno automaticamente.")
                game.make_move(None)
                time.sleep(1)
                continue

            print("Jogadas válidas (linha, coluna):")
            print([ (r+1, c+1) for (r,c) in moves ])

            while True:
                try:
                    s = input(f"Sua jogada ({player}) - digite 'linha coluna' (ex: 3 4): ").strip()
                    r, c = map(int, s.split())
                    move = (r-1, c-1)
                    if move in moves:
                        game.make_move(move)
                        break
                    else:
                        print("Jogada inválida. Escolha uma das jogadas válidas mostradas.")
                except Exception:
                    print("Entrada inválida. Use o formato: número_da_linha número_da_coluna.")
        # IA
        else:
            print("IA pensando...\n")
            if algorithm == "mcts":
                move = mcts(game, iterations=500)
            elif algorithm == "minimax":
                move = best_move_minimax_othello(game, depth=4)
            else:
                raise ValueError(f"Algoritmo desconhecido: {algorithm}")
            print(f"IA joga em: {move}")
            game.make_move(move)
            time.sleep(1)

    print_board_othello(game.board)
    x, o = game.score()
    print(f"Placar final -> {Fore.RED}X: {x}{Style.RESET_ALL} | {Fore.YELLOW}O: {o}{Style.RESET_ALL}")
    winner = game.winner()
    if winner == player:
        print(Fore.GREEN + "Você venceu!")
    elif winner == opp:
        print(Fore.RED + "A IA venceu.")
    else:
        print(Fore.CYAN + "Empate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Othello (Reversi) com IA colorida (MCTS / Minimax).")
    parser.add_argument("--algo", "-a", choices=["mcts", "minimax"], required=True,
                        help="Escolha o algoritmo da IA (mcts ou minimax).")
    args = parser.parse_args()
    play(args.algo)
