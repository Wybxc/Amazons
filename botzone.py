"""用于botzone的本地AI。"""

import asyncio
from typing import Optional, Union

import httpx
import yaml

from game import *
from play_demo import print_game


class BotzoneOnlineGame():
    """Botzone 在线对局。"""
    def __init__(self, uid: str, token: str):
        """
        Args:
            uid: 用户ID
            token: 密钥
        """
        self.uid = uid
        self.token = token

    async def start(self, botid: str, white: bool = False) -> str:
        """开始对局。
        Args:
            botid: 对战的机器人ID
            white: 自己是否为白棋
        """
        url = f'https://www.botzone.org/api/{self.uid}/{self.token}/runmatch'

        headers = {'X-Game': 'Amazons'}
        if not white:
            headers = {**headers, 'X-Player-0': 'me', 'X-Player-1': botid}
        else:
            headers = {**headers, 'X-Player-0': botid, 'X-Player-1': 'me'}

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.text

    async def request(
        self,
        game_id: str,
        response: Optional[str] = None,
    ) -> Union[str, bool]:
        """对局请求。
        Args:
            game_id: 对局ID
            response: 上一回合的响应

        Returns:
            如果游戏结束，返回True，否则返回请求。
        """
        url = f'https://www.botzone.org/api/{self.uid}/{self.token}/localai'

        async with httpx.AsyncClient(follow_redirects=True) as client:
            if response is None:
                resp = await client.get(url)
            else:
                resp = await client.get(
                    url, headers={f'X-Match-{game_id}': response}
                )
            resp.raise_for_status()

            status, *games = resp.text.split('\n')
            running, _ = status.split()
            if running == '1':
                return games[1]

            elif running == '0':
                _, color, _, p1, p2 = games[0].split()
                if color == '0':
                    return p1 == '2'
                else:
                    return p2 == '2'

            raise RuntimeError('More than one game found.')

    async def run(self, game_id: str, draw: bool = True) -> bool:
        """运行对局。
        Args:
            game_id: 对局ID
            draw: 是否绘制图像
        """
        game = Game(8)

        req = await self.request(game_id)
        i = 0
        white = True
        while isinstance(req, str):
            from_x, from_y, to_x, to_y, arrow_x, arrow_y = map(
                int, req.split()
            )
            if from_x >= 0:
                game.move(
                    Block(from_x, from_y), Block(to_x, to_y),
                    Block(arrow_x, arrow_y)
                )
            else:
                white = False

            if draw:
                print_game(game, True, i)

            try:
                i, _, from_pos, to_pos, arrow_pos = step(game, TimeLimit(50))
            except GameFinished as gg:
                if draw:
                    print_game(game, False, i)
                    print('Game finished!')
                    print('Winner:', 'white' if gg.winner else 'black')
                return gg.winner == white

            response = ' '.join(map(str, [*from_pos, *to_pos, *arrow_pos]))
            req = await self.request(game_id, response)
        return req

    async def play(
        self, botid: str, white: bool = False, draw: bool = True
    ) -> bool:
        """运行对局。
        Args:
            botid: 对战的机器人ID
            white: 自己是否为白棋
        """
        game_id = await self.start(botid, white)
        if draw:
            print('Me color:', 'white' if white else 'black')
        return await self.run(game_id, draw)


def play(white: bool):
    """开始本地 AI 对局。
    Args:
        white: 己方是否为白棋。
    """
    with open('botzone.yml') as f:
        config = yaml.safe_load(f)

    botid = config['botid']

    online_game = BotzoneOnlineGame(uid=config['uid'], token=config['token'])
    win = asyncio.run(online_game.play(botid, white, draw=True))
    print('Win!' if win else 'Lose!')


if __name__ == '__main__':
    t = time.time()
    while True:
        play(True)        
        if time.time() - t > 60:
            break
        else:
            time.sleep(20)
