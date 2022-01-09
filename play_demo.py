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
        steps: 搜索步数
    """
    print("\033c", end="")
    board_size = game.board_size
    print('Turn:', game.turn, 'Current:', 'White' if game.current else 'Black')
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

        this, other = mobility(game)
        m = (this+1e-10) / (this+other+2e-10)
        print(
            'White score: t1: {0:.2f}, t2: {2:.2f}, p1: {1:.2f}, p2: {3:.2f}, m: {4:.2f}'
            .format(*terroity_position(game), m)
        )
        print('Judge(Current): ', judge(game))
        print('Search steps:', steps)
        print('Search depth:', steps / (max(white_moves, black_moves) + 1))


def play():
    print(LOGO)

    board_size = 8

    game = Game(board_size)
    i = 0
    while True:
        print_game(game, True, i)
        limit = TimeLimit(5.5)
        try:
            i, game, *_ = step(game, limit)
        except GameFinished as gg:
            print_game(game, True, i)
            print('Winner:', 'white' if gg.winner else 'black')
            break


if __name__ == '__main__':
    play()