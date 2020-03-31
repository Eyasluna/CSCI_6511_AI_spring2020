import requests
import pandas as pd
import numpy as np
from copy import deepcopy

EMPTY = "-"
X = 'X'
O = 'O'
INF = float('inf')

player = X
oppo = O

N = 4
M = 3


class Player:

    def __init__(self, symbol):
        self.symbol = symbol
        self.oppo = X if symbol == O else X


class Game:

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.board = pd.DataFrame([[EMPTY for _ in range(n)] for _ in range(n)])

    def move(self, actions):
        for (row, col, t) in actions:
            self.board.loc[row, col] = t

    def check_win(self):
        for j in range(self.n):
            for i in range(self.n-self.m+1):

                s = self.board.loc[j, i:i+self.m-1]
                count = s.value_counts()
                if len(count) == 1 and count.index[0] != EMPTY:
                    return count.index[0]

                s = self.board.loc[i:i+self.m-1, j]
                count = s.value_counts()
                if len(count) == 1 and count.index[0] != EMPTY:
                    return count.index[0]

        for i in range(self.n-self.m+1):
            for j in range(self.n-self.m+1):
                s = pd.Series(np.diag(self.board.loc[i:i+self.m-1, j:j+self.m-1]))
                count = s.value_counts()
                if len(count) == 1 and count.index[0] != EMPTY:
                    return count.index[0]

                df = self.board.iloc[:, ::-1]
                df.columns = self.board.columns
                s = pd.Series(np.diag(df.loc[i:i + self.m - 1, j:j + self.m - 1]))
                count = s.value_counts()
                if len(count) == 1 and count.index[0] != EMPTY:
                    return count.index[0]

        return None

    def available_actions(self):
        actions = []
        for i in range(self.n):
            for j in range(self.n):
                if self.board.loc[i, j] == EMPTY:
                    actions.append((i, j))
        return actions

    def apply_action(self, action):
        new_game = deepcopy(self)
        new_game.move([action])
        return new_game

    def __str__(self):
        return str(self.board)


def make_new_game(n, m, actions):
    if actions is None:
        actions = []
    game = Game(n, m)
    game.move(actions)
    return game


def max_value(game, alpha, beta, depth):
    if depth == 0 or game.check_win():
        return evaluate(game)
    v = -INF
    for action in game.available_actions():
        i, j = action
        v = max(v, min_value(game.apply_action([i, j, player]), alpha, beta, depth - 1))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v


def min_value(game, alpha, beta, depth):
    if depth == 0 or game.check_win():
        return evaluate(game)
    v = INF
    for action in game.available_actions():
        i, j = action
        v = min(v, max_value(game.apply_action([i, j, oppo]), alpha, beta, depth-1))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v


def check_score(s, game):
    s = s.to_dict()
    on = s[oppo] if oppo in s else 0
    pn = s[player] if player in s else 0
    if on == game.m:
        return -INF
    if pn == game.m:
        return INF

    if on > 0:
        if pn == 0:
            return -(game.n ** on)
        else:
            return 0
    else:
        return game.n ** pn


def evaluate(game):
    score = 0
    for j in range(game.n):
        for i in range(game.n - game.m + 1):

            s = game.board.loc[j, i:i + game.m - 1]
            count = s.value_counts()
            score += check_score(count, game)

            s = game.board.loc[i:i + game.m - 1, j]
            count = s.value_counts()
            score += check_score(count, game)

    for i in range(game.n - game.m+1):
        for j in range(game.n - game.m+1):
            s = pd.Series(np.diag(game.board.loc[i:i + game.m - 1, j:j + game.m - 1]))
            count = s.value_counts()
            score += check_score(count, game)

            s = pd.Series(np.diag(game.board.iloc[:, ::-1].loc[i:i + game.m - 1, j:j + game.m - 1]))
            count = s.value_counts()
            score += check_score(count, game)
    return score


def alpha_beta(game, max_depth=5):

    best_score = -INF
    best_action = None

    for action in game.available_actions():
        i, j = action
        v = min_value(game.apply_action([i, j, player]), best_score, INF, max_depth-1)
        # print(action, v)
        if v > best_score:
            best_score = v
            best_action = [i, j, player]

    return best_action


def human_vs_computer():
    game = make_new_game(N, M, None)
    while game.check_win() is None:
        action = alpha_beta(game, 3)
        game = game.apply_action([action[0], action[1], player])
        print(game)

        if game.check_win():
            break

        action = input("input row, col: ").split(" ")
        game = game.apply_action([int(action[0]), int(action[1]), oppo])
        print(game)

    print("end ")
    print(game)


p1 = {
    "function": alpha_beta,
    "player": Player(O),
    "USERID": "930",
    "API_KEY": "31dc33beb3873e2bd595",
    "TEAM_ID": "1223"
}
p2 = {
    "function": alpha_beta,
    "player": Player(X),
    "USERID": "845",
    "API_KEY": "a66101f392d27f42c4cd",
    "TEAM_ID": "1227"
}


def computer_vs_computer(player1, player2):
    gameId = create_a_new_game(player1, player2)
    print("gameID: ", gameId)
    while True:
        game = get_game(gameId, player1)
        action = player1['function'](game, 1)
        print(action)
        make_a_move(gameId, (action[0], action[1]), player1)
        game = get_game(gameId, player1)
        print(game)
        if game.check_win():
            print("player1 win")
            break

        game = get_game(gameId, player2)
        action = player2['function'](game, 1)
        print(action)
        make_a_move(gameId, (action[0], action[1]), player2)
        game = get_game(gameId, player2)
        print(game)
        if game.check_win():
            print("player2 win")
            break



import json
url = "https://www.notexponential.com/aip2pgaming/api/index.php"


def create_a_new_game(player1, player2):
    teamId1, teamId2 = player1["TEAM_ID"], player2["TEAM_ID"]
    USERID = player1["USERID"]
    API_KEY = player1["API_KEY"]
    import http.client
    conn = http.client.HTTPSConnection("www.notexponential.com")

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; " \
              "name=\"type\"\r\n\r\ngame\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; " \
              "name=\"gameType\"\r\n\r\nTTT\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; " \
              "name=\"teamId1\"\r\n\r\n"+teamId1+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; " \
                                                 "name=\"teamId2\"\r\n\r\n"+teamId2+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': USERID,
        'x-api-key': API_KEY,
        'cache-control': "no-cache",
        'postman-token': "41d5f39d-0533-74b4-4c35-9656b8c539bd"
    }

    conn.request("POST", "/aip2pgaming/api/index.php", payload, headers)

    res = conn.getresponse()
    data = res.read()

    s = data.decode("utf-8")
    return json.loads(s)["gameId"]

def create_a_new_game2(player1, player2):
    """return game id"""
    teamId1, teamId2 = player1["TEAM_ID"], player2["TEAM_ID"]
    USERID = player1["USERID"]
    API_KEY = player1["API_KEY"]
    print(teamId1,teamId2, USERID, API_KEY)
    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ngame\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameType\"\r\n\r\nTTT\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId1\"\r\n\r\n"+teamId1+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId2\"\r\n\r\n"+teamId2+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    # payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\ngame\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameType\"\r\n\r\nTTT\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId1\"\r\n\r\n1223\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId2\"\r\n\r\n1227\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    'authorization': "Basic Og==",
    'userid': "930",
    'x-api-key': "31dc33beb3873e2bd595",
    'cache-control': "no-cache",
    'postman-token': "d309f41d-1c1e-faa4-1ac5-20c6e4b6fb1c"
    }

    response = requests.request("POST", url, data=payload, headers=headers)
    if str(response.status_code).startswith("4"):
        print("server error!")

    print(response.text)
    return json.loads(response.text)["gameId"]


def get_game(gameId, player):
    import http.client

    USERID = player["USERID"]
    API_KEY = player["API_KEY"]

    conn = http.client.HTTPSConnection("www.notexponential.com")

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nboardString\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameId\"\r\n\r\n"+str(gameId)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': USERID,
        'x-api-key': API_KEY,
        'cache-control': "no-cache",
        'postman-token': "75910fd5-a16c-ed0a-96eb-047b73b841ed"
    }

    conn.request("GET", "/aip2pgaming/api/index.php?type=boardString&gameId="+str(gameId), payload, headers)

    res = conn.getresponse()
    data = res.read()

    s = data.decode("utf-8")

    res = json.loads(s)
    b = [l for l in res['output'].split("\n") if len(l) > 0]
    n = len(b)
    game = Game(n, res['target'])
    for i in range(n):
        line = b[i]
        for j in range(n):
            game.board.loc[i, j] = line[j]
    return game


def get_game2(gameId, player):
    url = "https://www.notexponential.com/aip2pgaming/api/index.php"

    querystring = {"type": "boardString", "gameId": str(gameId)}

    USERID = player["USERID"]
    API_KEY = player["API_KEY"]

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nboardString\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameId\"\r\n\r\n"+str(gameId)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': USERID,
        'x-api-key': API_KEY,
        'cache-control': "no-cache",
    }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)
    print(response)
    res = json.loads(response.text)
    n = len(res['output'])
    game = Game(n, res['target'])
    b = res['output'].split("\n")
    for i in range(n):
        line = b[i]
        for j in range(n):
            game.board.loc[i, j] = line[j]
    return game


def make_a_move(gameId, move, player):
    teamId = player["TEAM_ID"]
    USERID = player["USERID"]
    API_KEY = player["API_KEY"]

    import http.client

    conn = http.client.HTTPSConnection("www.notexponential.com")

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nmove\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameId\"\r\n\r\n"+str(gameId)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"move\"\r\n\r\n"+str(move[0])+","+str(move[1])+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId\"\r\n\r\n"+teamId+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"

    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': USERID,
        'x-api-key': API_KEY,
        'cache-control': "no-cache",
        'postman-token': "50b6539a-f0de-8784-4a2a-2813322019ae"
    }

    conn.request("POST", "/aip2pgaming/api/index.php", payload, headers)

    res = conn.getresponse()
    data = res.read()

    s = data.decode("utf-8")
    print(s)

    return json.loads(s)


def make_a_move2(gameId, move, player):
    teamId = player["TEAM_ID"]
    USERID = player["USERID"]
    API_KEY = player["API_KEY"]

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nmove\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameId\"\r\n\r\n"+str(gameId)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"move\"\r\n\r\n"+str(move)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"teamId\"\r\n\r\n"+str(teamId)+"\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': USERID,
        'x-api-key': API_KEY,
        'cache-control': "no-cache",
    }

    response = requests.request("POST", url, data=payload, headers=headers)

    return json.loads(response.text)


def get_the_move_list(gameId, count, player1, player2):


    querystring = {"type": "moves", "gameId": str(gameId), "count": str(count)}

    payload = "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"type\"\r\n\r\nboardString\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW\r\nContent-Disposition: form-data; name=\"gameId\"\r\n\r\n227\r\n------WebKitFormBoundary7MA4YWxkTrZu0gW--"
    headers = {
        'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
        'authorization': "Basic Og==",
        'userid': "392",
        'x-api-key': "430b12204290d33a0edc",
        'cache-control': "no-cache",
        'postman-token': "d4851264-e838-9c7d-e3e6-5ee9e22b198d"
    }

    response = requests.request("GET", url, data=payload, headers=headers, params=querystring)

    # print(response.text)
    return json.loads(response.text)


if __name__ == '__main__':
    # human_vs_computer()
    computer_vs_computer(p1, p2)
    # df = pd.read_csv("1.csv", header=None)
    # game = Game(12, 6)
    # game.board = df
    # print(game.check_win())