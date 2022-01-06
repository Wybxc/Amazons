"""亚马逊棋游戏规则实现. 
"""
import functools
import time
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, TypeVar

import numpy as np


class Block(NamedTuple):
    x: int
    y: int

    def __add__(self, other: 'Block'):
        return Block(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Block'):
        return Block(self.x - other.x, self.y - other.y)


# 上, 左, 左上, 右上, 下, 右, 右下, 左下
_DX = [0, -1, -1, 1, 0, 1, 1, -1]
_DY = [-1, 0, -1, -1, 1, 0, 1, 1]

T = TypeVar('T')


def hash_cache(func: Callable[['Game'], T]) -> Callable[['Game'], T]:
    _hash: Dict[int, T] = {}

    @functools.wraps(func)
    def wrapper(game: Game) -> T:
        hash_key = hash(game)
        if hash_key not in _hash:
            _hash[hash_key] = func(game)
        return _hash[hash_key]

    return wrapper


class Game():
    """亚马逊棋游戏. """
    board: np.ndarray
    """棋盘, 0空地, 1障碍, 2白棋, 3黑棋. """
    board_size: int
    """棋盘大小. """
    amazons_white: List[Block]
    """白方的四个亚马逊. """
    amazons_black: List[Block]
    """黑方的四个亚马逊. """
    current: bool
    """当前棋子颜色. True 表示白方, False 表示黑方. """
    turn: int
    """回合数."""
    def __init__(self, board_size: int, fork: bool = False):
        self.board_size = board_size
        if not fork:
            self._reset()

    def _reset(self):
        """初始化棋盘. """
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.int8
        )

        bs13 = self.board_size // 3
        bs23 = self.board_size - bs13 - 1
        bsm = self.board_size - 1
        self.amazons_white = [
            Block(0, bs23),
            Block(bs13, bsm),
            Block(bs23, bsm),
            Block(bsm, bs23),
        ]
        self.amazons_black = [
            Block(0, bs13),
            Block(bs13, 0),
            Block(bs23, 0),
            Block(bsm, bs13),
        ]

        for white in self.amazons_white:
            self.board[white] = 2
        for black in self.amazons_black:
            self.board[black] = 3

        self.current = False  # 黑方先行
        self.turn = 0

    def fork(self):
        """获取局面的一个副本。"""
        new_game = Game(self.board_size, fork=True)
        new_game.board = self.board.copy()
        new_game.amazons_white = self.amazons_white.copy()
        new_game.amazons_black = self.amazons_black.copy()
        new_game.current = self.current
        new_game.turn = self.turn
        return new_game

    def __len__(self):
        # return 4 * 4 * (self.board_size - 1) * 4 * (self.board_size - 1)
        return 64 * (self.board_size - 1) * (self.board_size - 1)

    def amazon(self, color: Optional[bool] = None):
        """获取当前棋子颜色的亚马逊. """
        if color is None:
            color = self.current
        return self.amazons_white if color else self.amazons_black

    def __hash__(self):
        return hash(self.board.data.tobytes())

    def decode_action(
        self,
        action: int,
        color: Optional[bool] = None
    ) -> Tuple[int, Block, Block]:
        """将网络输出的动作解码为棋盘上的位置. 
        
        网络输出: [0, 4 * 4 * (board_size-1) * 4 * (board_size-1)) 内的整数. 
        其中 4 表示 4 个亚马逊之一, 4 * (board_size-1) 表示一次移动. 
        4 表示移动方向, 按照上、左、左上、右上的顺序编码. 
        (board_size-1) 表示移动距离, 以 x 表示正向移动, (board_size-1)-x 表示逆向移动. 

        Args:
            action: 网络输出的动作. 
            color: 棋子颜色, True 表示白方, False 表示黑方. 
        """
        amazons = self.amazon(color)

        # unpack
        action, a_distance = divmod(action, self.board_size - 1)
        action, a_direction = divmod(action, 4)
        action, m_distance = divmod(action, self.board_size - 1)
        action, m_direction = divmod(action, 4)
        index = action

        x, y = amazons[index]

        dx, dy = _DX[m_direction], _DY[m_direction]
        x += dx * (m_distance+1)
        y += dy * (m_distance+1)
        if not (0 <= x < self.board_size):
            x -= dx * self.board_size
        if not (0 <= y < self.board_size):
            y -= dy * self.board_size
        move = Block(x, y)

        dx, dy = _DX[a_direction], _DY[a_direction]
        x += dx * (a_distance+1)
        y += dy * (a_distance+1)
        if not (0 <= x < self.board_size):
            x -= dx * self.board_size
        if not (0 <= y < self.board_size):
            y -= dy * self.board_size
        arrow = Block(x, y)

        return index, move, arrow

    def valid_actions(self, color: Optional[bool] = None) -> np.ndarray:
        """获取当前局面下所有合法的动作.
        
        Returns:
            np.ndarray[bool] 动作合法时为 True, 否则为 False.
        """
        result = np.zeros(len(self), dtype=np.bool_)
        color = color if color is not None else self.current
        amazons = self.amazon(color)
        for index, amazon in enumerate(amazons):
            x0, y0 = amazon
            for m_direction in range(8):
                for m_distance in range(7):
                    x1 = x0 + _DX[m_direction] * (m_distance+1)
                    y1 = y0 + _DY[m_direction] * (m_distance+1)
                    if not (
                        0 <= x1 < self.board_size and 0 <= y1 < self.board_size
                    ):
                        break
                    if self.board[x1, y1]:
                        break

                    m_dir = m_direction
                    m_dist = m_distance
                    if m_direction >= 4:
                        m_dir -= 4
                        m_dist = self.board_size - 2 - m_distance
                    for a_direction in range(8):
                        for a_distance in range(7):
                            x2 = x1 + _DX[a_direction] * (a_distance+1)
                            y2 = y1 + _DY[a_direction] * (a_distance+1)
                            if not (
                                0 <= x2 < self.board_size
                                and 0 <= y2 < self.board_size
                            ):
                                break
                            if self.board[x2, y2] and amazon != (x2, y2):
                                break

                            a_dir = a_direction
                            a_dist = a_distance
                            if a_direction >= 4:
                                a_dir -= 4
                                a_dist = self.board_size - 2 - a_distance

                            # pack
                            action = index
                            action = action*4 + m_dir
                            action = action * (self.board_size - 1) + m_dist
                            action = action*4 + a_dir
                            action = action * (self.board_size - 1) + a_dist
                            result[action] = True

        return result.flatten()

    def check_obstacle(
        self, from_pos: Block, to_pos: Block, free: Optional[Block] = None
    ):
        p = to_pos - from_pos
        if p.x != 0 and p.y != 0 and abs(p.x) != abs(p.y):
            return False

        dx, dy = np.sign(p.x), np.sign(p.y)
        while from_pos != to_pos:
            from_pos = from_pos + Block(dx, dy)
            if self.board[from_pos] and from_pos != free:
                return False

        return True

    def is_valid_move(
        self, from_pos: Block, to_pos: Block, color: Optional[bool] = None
    ):
        """检查棋子移动是否合法。"""
        if not (
            0 <= from_pos.x < self.board_size
            and 0 <= from_pos.y < self.board_size and
            0 <= to_pos.x < self.board_size and 0 <= to_pos.y < self.board_size
        ):
            return False

        amazons = self.amazon(color)
        if from_pos not in amazons:
            return False

        if self.board[to_pos]:
            return False

        if not self.check_obstacle(from_pos, to_pos):
            return False

        return True

    def is_valid_arrow(self, from_pos: Block, to_pos: Block, arrow_pos: Block):
        """检查安放障碍是否合法。此方法假设 from_pos 和 to_pos 已经是合法的。"""
        if not (
            0 <= arrow_pos.x < self.board_size
            and 0 <= arrow_pos.y < self.board_size
        ):
            return False

        if to_pos == arrow_pos:
            return False

        if self.board[arrow_pos] and arrow_pos != from_pos:
            return False

        if not self.check_obstacle(to_pos, arrow_pos, free=from_pos):
            return False

        return True

    def _move(self, index: int, to_pos: Block, arrow_pos: Block):
        """进行棋子移动（不检查合法性）。"""
        amazons = self.amazon()

        self.board[to_pos] = 2 if self.current else 3
        self.board[amazons[index]] = 0

        self.board[arrow_pos] = 1

        amazons[index] = to_pos

        self.current = not self.current
        self.turn += 1

        return self

    def move(self, from_pos: Block, to_pos: Block, arrow_pos: Block):
        """进行棋子移动（不检查合法性）。"""
        amazons = self.amazon()
        return self._move(amazons.index(from_pos), to_pos, arrow_pos)

    def do_action(
        self,
        action: int,
    ):
        """按照网络输出的动作进行移动."""
        index, move, arrow = self.decode_action(action)
        return self._move(index, move, arrow)


@hash_cache
def valid_actions_num(game: Game) -> Tuple[int, int]:
    """获取当前局面下所有合法的动作的数目.
    
    Returns:
        白棋, 黑棋.
    """
    arrow_count = np.full(
        (game.board_size, game.board_size, 8), -1, dtype=np.int32
    )

    def get_arrow_count(i: int, j: int, d: int):
        if arrow_count[i, j, d] == -1:
            tx, ty = i + _DX[d], j + _DY[d]
            if not (
                0 <= tx < game.board_size and 0 <= ty < game.board_size
                and not game.board[tx, ty]
            ):
                arrow_count[i, j, d] = 0
            else:
                arrow_count[i, j, d] = 1 + get_arrow_count(tx, ty, d)
        return arrow_count[i, j, d]

    def get_move_count(p: Block):
        free = [1] * 8  # 由于棋子移动, 在d方向上多出来的可以放置障碍的空位数量
        for d in range(8):
            tx, ty = p.x + _DX[d], p.y + _DY[d]
            while (
                0 <= tx < game.board_size and 0 <= ty < game.board_size
                and not game.board[tx, ty]
            ):
                free[d] += 1
                tx, ty = tx + _DX[d], ty + _DY[d]

        result = 0
        for d in range(8):
            tx, ty = p.x + _DX[d], p.y + _DY[d]
            while (
                0 <= tx < game.board_size and 0 <= ty < game.board_size
                and not game.board[tx, ty]
            ):
                for dd in range(8):
                    result += get_arrow_count(tx, ty, dd)
                    if d - dd == 4 or d - dd == -4:
                        result += free[dd]
                tx, ty = tx + _DX[d], ty + _DY[d]
        return result

    white_num, black_num = 0, 0
    for white in game.amazons_white:
        white_num += get_move_count(white)
    for black in game.amazons_black:
        black_num += get_move_count(black)
    return white_num, black_num


# def queen_king_move(self):


def judge(game: Game) -> float:
    """给出当前局面的胜率评估。"""
    color = game.current

    this, other = valid_actions_num(game)
    if not color:
        other, this = this, other

    score = (this+1e-10) / (this+other+2e-10)

    return float(score)


class GameFinished(Exception):
    """游戏结束。"""
    def __init__(self, winner: bool):
        """
        Args:
            winner: 游戏胜利者, True 表示白方胜利, False 表示黑方胜利.
        """
        self.winner = winner


class MCTNode():
    """蒙特卡洛树节点.
    包含一个游戏局面，以及参数 N(访问次数), Q(总价值), P(先验概率).
    """
    game: Game
    """当前的游戏局面."""
    children: List['MCTNode']
    """子节点."""
    actions: List[int]
    """子节点对应的动作."""
    N: int
    """节点的访问次数."""
    Q: float
    """节点的总价值."""
    base_score: float
    """父节点的分数."""
    def __init__(self, game: Game, base_score: float):
        self.game = game
        self.children = []
        self.actions = []
        self._available_actions: Optional[Set[int]] = None
        self.N = 0
        self.Q = 0
        self.base_score = base_score

    @property
    def available_actions(self) -> Set[int]:
        """当前局面未选择的合法动作."""
        if self._available_actions is None:
            self._available_actions = set(
                np.where(self.game.valid_actions())[0]
            )
        return self._available_actions

    def UCT(self, exploration: float) -> Tuple[np.ndarray, int]:
        """计算子节点的UCT值.
        
        Args:
            exploration: 探索系数, 介于0和1之间.

        Returns:
            UCT值, 对应的动作在节点中的序号.
        """
        N = np.empty(len(self.children), dtype=np.float32)
        Q = np.empty(len(self.children), dtype=np.float32)
        for i, child in enumerate(self.children):
            N[i] = 1e-10 + child.N
            Q[i] = child.Q

        uct = Q/N + exploration * np.sqrt(2 * np.log(self.N) / N)
        index = uct.argmax()

        return uct, index

    def ended(self):
        """判断当前局面是否结束."""
        return len(self.available_actions) == 0 and len(self.actions) == 0

    def expandable(self):
        """检查是否可以展开子节点."""
        return len(self.available_actions) > 0

    def expand(self):
        """扩展子节点."""
        action = self.available_actions.pop()
        new_game = self.game.fork().do_action(action)
        new_node = MCTNode(new_game, judge(self.game))

        self.children.append(new_node)
        self.actions.append(action)

        return new_node

    def rollout(self) -> float:
        """计算当前局面的收益."""
        return judge(self.game) - self.base_score


class MCT():
    """蒙特卡洛树."""
    board_size: int
    """棋盘大小."""
    root: MCTNode
    """根节点."""
    def __init__(self, game: Game):
        self.board_size = game.board_size

        self.root = MCTNode(game, 0.)
        self.root.N = 1

    def search(self):
        """进行一次蒙特卡洛树搜索."""
        path, node = self.select()
        self.backup(path, node)
        if self.root.ended():
            raise GameFinished(not self.root.game.current)

    def policy(self):
        """根据搜索结果, 计算下一步操作."""
        _, index = self.root.UCT(0)
        return self.root.actions[index]

    def select(self):
        """选择一个子节点, 返回选择路径上的所有节点.
        
        Returns:
            path: 路径上节点的列表, 不包括选择的节点.
            node: 选择的节点.
        """
        node = self.root
        path: List[MCTNode] = []

        while not node.ended():
            path.append(node)
            if node.expandable():
                return path, node.expand()
            else:
                _, next_index = node.UCT(1)
                node = node.children[next_index]

        return path, node

    def backup(self, path: Sequence[MCTNode], node: MCTNode):
        """向上传播."""
        delta = -node.rollout()

        node.N += 1
        node.Q += delta
        delta = -delta

        for pnode in reversed(path):
            pnode.N += 1
            pnode.Q += delta
            delta = -delta


def step(game: Game, limit: Callable[[], bool]):
    mct = MCT(game)

    i = 0
    while limit():
        i += 1
        mct.search()

    action = mct.policy()
    index, to_pos, arrow_pos = game.decode_action(action)
    from_pos = game.amazon()[index]

    # assert game.is_valid_move(from_pos, to_pos)
    # assert game.is_valid_arrow(from_pos, to_pos, arrow_pos)
    game.do_action(action)

    return i, game, from_pos, to_pos, arrow_pos


class TimeLimit():
    """时间限制."""
    def __init__(self, t: float):
        self._t = t
        self._start = time.time()

    def __call__(self):
        return time.time() - self._start < self._t


class CountLimit():
    """计数限制."""
    def __init__(self, n: int):
        self._n = n
        self._count = 0

    def __call__(self):
        self._count += 1
        return self._count <= self._n


if __name__ == "__main__":
    game = Game(8)
    turn = int(input())
    for i in range(2*turn - 1):
        from_x, from_y, to_x, to_y, arrow_x, arrow_y = map(
            int,
            input().split()
        )
        if from_x >= 0:
            game.move(
                Block(from_x, from_y), Block(to_x, to_y),
                Block(arrow_x, arrow_y)
            )

    _, _, from_pos, to_pos, arrow_pos = step(game, TimeLimit(5.5))
    print(' '.join(map(str, [*from_pos, *to_pos, *arrow_pos])))