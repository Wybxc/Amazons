# 亚马逊棋

本项目为北京大学计算概论A（实验班）的课程作业，使用蒙特卡洛树搜索算法编写的亚马逊棋AI程序。

Botzone ID：61daede099f5414277347668

## 项目结构

### game.py

游戏规则、AI算法的实现部分，可直接在 botzone 平台上运行。

### play_demo.py

自对弈演示。运行后，开始一场自对弈，在控制台界面显示棋盘，并实时显示估值函数的结果以及蒙特卡洛树的搜索深度。

### botzone.py

用于 botzone 的本地 AI 功能，运行后将和指定 ID 的 bot 开启一场比赛。默认从当前文件夹下的 `botzone.yml`读取信息（出于隐私保护原因，此文件未添加到代码仓库中）。

## 环境依赖

开发环境为 Python 3.8.11，Windows 10。`game.py` 和 `play_demo.py` 兼容 Python 3.6+，`botzone.py`兼容 Python 3.7+。

`game.py `和 `play_demo.py` 只依赖于 `numpy`。`botzone.py` 同时依赖 `httpx` 库。

## 搜索算法

搜索算法采用变种蒙特卡洛树搜索，**沿用了 UCT 算法，取消了蒙特卡洛树中的 rollout，改为用估值函数取代**。

### 蒙特卡洛树搜索

蒙特卡洛树搜索（MCTS）部分主要涉及 `game.py` 的 `MCTNode` 和 `MCT` 类。

回合开始时，首先将当前局面创建为 MCT 的根节点。然后从根节点开始，在树上执行若干次拓展-搜索操作，直到达到时限，返回最优结果作为该回合的行动。

拓展-搜索操作分为选择和回溯两个步骤。在选择步骤（见`MCT.select`方法）中，采用 UCT 算法（Kocsis L, Szepesvári C, 2006）作为依据，从根节点开始，不断进入 UCT 值最大的子节点，直到找到子节点尚未探索完全的节点，并随机选择一个子节点加入树中，将新的节点作为选择的结果；如果树已经完全展开，则选择 UCT 值最大的叶子节点。

在回溯步骤（见`MCT.backup`方法）中，利用估值函数计算选择的节点的价值，并依据此向上更新选择路径上所有节点的总价值和访问次数。**受 AlphaZero 的启发，为加快计算速度，不使用 rollout 算法进行向后随机模拟，而是直接将节点对应局面输入估值函数中，得到局面的评分作为节点的价值**。

MCTS 部分的伪代码如下：

```python
def mcts(game: Game):
    root = MCTNode(game=game)
    while not Timeout:
        node = select(root)
        backup(node)
    return best_child(root, 0).action

# 选择
def select(node: MCTNode) -> MCTNode:
    while not node.game.finished():
	    if len(node.children) < node.max_children_count:
            child = node.new_child()
            node.children.append(child)
            return child
        else:
            node = best_child(node, 1)
 	return node

def best_child(node: MCTNode, ε: float):
	return max(node.children, key=UCT(ε))

def UCT(ε: float):
    def f(node: MCTNode):
        Q, N = node.Q, node.N
        return Q / N + ε * sqrt(2 * ln(node.parent.N) / N)
    return f

# 回溯
def backup(node: MCTNode):
    δ = evaluate(node.game)
    while node is not root:
        node.Q += δ
        node.N += 1
        node = node.parent
        δ = -δ

# 估值函数
def evaluate(game: Game):
    ...  
```

### 估值函数

估值函数使用了 territory、position、mobility 三种特征值（Lieberum J, 2005），并按照游戏进程，对三种特征值进行加权（郭琴琴,李淑琴,包华, 2012）。

首先，定义 Queen move 函数 $D_1^i(p)$为玩家 $i$ 的亚马逊按照皇后走法，走到空地格 $p$ 所需要的最少步数，以及 King move 函数 $D_2^i(p)$为玩家 $i$ 的亚马逊按照皇后走法，走到空地格 $p$ 所需要的最少步数。

那么，玩家 1 的 territory 特征值可计算如下：
$$
t_i=\sum_{p \in \textrm{Empty}} \Delta(D_i^1(p), D_i^2(p))\\
\textrm{where}\quad \Delta(a,b)=\begin{cases}
0, & a=b=\infty\\
\kappa, &a=b<\infty\\
1, & a<b\\
-1, & a>b
\end{cases}
$$
其中 $\kappa$ 为先手优势，取 $\kappa=0.2$。

玩家 1 的 position 特征值可计算如下：
$$
p_1=\sum_{p \in \textrm{Empty}} \left(2^{1-D_1^1(p)}-2^{1-D_1^2(p)}\right)\\
p_2=\sum_{p \in \textrm{Empty}} \mathrm{clip}\left(\frac{D_2^2(p)-D_2^1(p)}{6}, -1, 1\right)\\[1em]
\textrm{where}\quad \mathrm{clip}(a, l, h)=\min(h, \max(l, a))
$$
定义 $M_i(k)$ 为玩家 $i$ 当前状态下，序号为 $k$ 的亚马逊的可能行动数，计算玩家 1 的 mobility 特征值：
$$
m=\frac{M_1+\varepsilon}{M_1+M_2+2\varepsilon} \\
\textrm{where}\quad M_i=\sum_{1\le k\le 4} M_i(k) + 4\min_{1\le k\le 4}M_i(k)\\
$$
其中 $\varepsilon=10^{-10}$ 是为了防止分母为0添加的修正项。

注意到对手的 territory 特征值与 position 特征值恰与我方的对应值为相反数，因此采用 sigmoid 函数进行归一化：
$$
\sigma(x)=\frac{1}{1+e^{-2x}}\\
t_i'=\sigma(t_i)\\
p_i'=\sigma(p_i)
$$
归一化之后的各项特征值均在 $[0, 1)$ 之内，按照一定的权重进行加权，得到最终的估值函数。

权重参考了 [3] 中的结论，由于 botzone 的棋盘较小，所以游戏进程更快，因此对原文献中权重调整的阈值进行了修改。修改后的权重如下：

| 回合数(n) | t_1  | t_2  | p_1  | p_2  | m    |
| --------- | ---- | ---- | ---- | ---- | ---- |
| 1~15      | 0.14 | 0.37 | 0.13 | 0.13 | 0.23 |
| 16~35     | 0.3  | 0.25 | 0.2  | 0.2  | 0.05 |
| 36~       | 0.8  | 0.1  | 0.05 | 0.05 | 0    |

综合估值函数的值域为 $[0, 1)$ 。为契合 MCTS，将平衡时的估值调整为 0，最终的估值函数经过线性函数映射到 $[-1,1]$ 上。
$$
q=\alpha t_1'+\beta t_2'+\gamma p_1'+\delta p_2' + \epsilon m\\
q'=2q-1
$$

## 实现细节

Python 运行速度较慢，这导致 MCTS 在一定时间内搜索次数不足，无法达到最好效果。在具体实现时，使用了一些技巧来提高运行速度，改善 MCTS 的表现。

使用 `pyinstrument` 对程序进行性能分析，找出耗时较长的函数，以进行针对式优化。分析结果见 `profile.html`。

根据性能分析，估值函数的计算是耗时最多的部分。

### 局面哈希

估值函数是关于局面的纯函数。使用哈希表储存计算过的估值，可以节省重复计算的时间，但同时也要求耗时极短的哈希算法。

对于使用 numpy 数组储存的局面，可以使用 `data` 属性访问底层内存。将数组视为字节流，利用内置的 `hash` 函数计算（见 `Game.__hash__`），经测试，这种方式的哈希运算速度可达到平均 $2.3\times 10^6$ 次每秒。

### 针对 Queen move 和 King move 的优化

性能分析中，发现 Queen move 和 King move 计算中采用的备忘录递归算法产生了很多不必要的函数调用开销。针对这一问题，在递归开始之前，进行一次迭代，计算出 Queen move 为 1 的点，以消除一些递归边界条件。同时，在递归内部进行一次手动的函数部分内联，减少函数调用次数。

对于 King move，基于两点之间直线最短的原理，可以套用 Queen move 的计算方式，只需将 +1 改为增加移动格数。这样可以起到剪枝的作用。

经过性能优化，Queen move 和 King move 的总计算耗时降低了 75%。

### 使用 numpy 向量运算的优化

利用 numpy 的向量运算的速度快于 Python 循环的速度，在计算 territory 和 position 特征值时，对全体空格求和的操作，可以利用 numpy 的向量运算实现，从而加速运行。

## 未解决的问题

开发中的一个版本（Botzone ID：61d8580e99f541427731edef）出现了异常好的表现，在某些情况下甚至可以超过最新版本。

目前已知此版本中存在一些实现上的错误：territory 特征值中先手优势的符号颠倒，以及使用了错误的差分算法估计局面价值。

此版本在存在错误的情况下拥有良好表现的原因，由于时间不足，目前尚未探明。在之后的版本中，修复了其中的符号错误，但此时差分算法反而降低了表现，同时因为差分算法理论不可解释，而删除了差分算法。

在下一步的研究中，比较此异常版本和最新版本的差异，从理论上加以分析，也许可以为进一步改进算法提供新的思路。

## 参考文献

[1] Kocsis L, Szepesvári C. Bandit based monte-carlo planning[C]//European conference on machine learning. Springer, Berlin, Heidelberg, 2006: 282-293.

[2] Lieberum J. An evaluation function for the game of amazons[J]. Theoretical computer science, 2005, 349(2): 230-244.

[3] 郭琴琴,李淑琴,包华.亚马逊棋机器博弈系统中评估函数的研究[J].计算机工程与应用,2012,48(34):50-54+87.



