"""自对弈演示.
"""

from game import *

LOGO = r"""
    ___                                          
   /   |  ____ ___  ____ _____  ____  ____  _____
  / /| | / __ `__ \/ __ `/_  / / __ \/ __ \/ ___/
 / ___ |/ / / / / / /_/ / / /_/ /_/ / / / (__  ) 
/_/  |_/_/ /_/ /_/\__,_/ /___/\____/_/ /_/____/  
"""

chess = {
    0: ' ',
    1: '●',
    2: '✧',
    3: '✦',
}


def print_game(game: Game, with_score: bool, steps: int):
    """打印棋盘。
    Args:
        game: 棋盘
        with_score: 是否打印得分
    """
    print("\033c", end="")
    board_size = game.board_size
    print('Current:', 'White' if game.current else 'Black')
    print('┏' + '━━━┳' * (board_size-1) + '━━━┓')
    for row in range(board_size):
        print('┃', end='')
        for col in range(board_size):
            print(f' {chess[game.board[col, row]]} ', end='')
            if col != board_size - 1:
                print('┃', end='')
        print('┃')
        if row != board_size - 1:
            print('┣' + '━━━╋' * (board_size-1) + '━━━┫')
    print('┗' + '━━━┻' * (board_size-1) + '━━━┛')

    if with_score:
        white_moves = int(game.valid_actions(True).sum())
        black_moves = int(game.valid_actions(False).sum())
        print(f'White: {white_moves:4}  Black: {black_moves:4}')
        white_score = judge(game) if game.current else 1 - judge(game)
        black_score = 1 - white_score
        print(f'White: {white_score:4f}  Black: {black_score:4f}')
        print('Search steps:', steps)
        print('Search depth:', steps / max(white_moves, black_moves))


def play():
    print(LOGO)

    board_size = 8

    game = Game(board_size)

    i = 0
    while True:
        print_game(game, True, i)
        limit = TimeLimit(1)
        try:
            i, game, *_ = step(game, limit)
        except GameFinished as gg:
            print_game(game, False, i)
            print('Game finished!')
            print('Winner:', 'white' if gg.winner else 'black')
            return


if __name__ == '__main__':
    play()