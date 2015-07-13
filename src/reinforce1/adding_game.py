#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

"""
問題： 以下の様なGameを強化学習？で学べるか
Game:
    状態S: 0~9 の整数
    アクションA: 1~4の整数
    状態S': (S + A) % 10
    報酬R:
        +1:  S' == 7
        -100: S' in (5, 9)

学習の評価：
    100回アクションした際の報酬Rの総和
"""

import numpy as np
from random import random, choice


class Game(object):
    state = None
    actions = []
    game_over = False

    def __init__(self, player):
        self.player = player
        self.turn = 0
        self.last_reward = 0
        self.total_reward = 0
        self.init_state()

    def player_action(self):
        action = self.player.action(self.state, self.last_reward)
        if action not in self.actions:
            raise Exception("Invalid Action: '%s'" % action)
        self.state, self.last_reward = self.get_next_state_and_reward(self.state, action)

    def play(self):
        yield(self)
        while not self.game_over:
            self.player_action()
            self.turn += 1
            self.total_reward += self.last_reward
            yield(self)

    def init_state(self):
        raise NotImplemented()

    def get_next_state_and_reward(self, state, action):
        raise NotImplemented()

class AddingGame(Game):
    """
    状態S: 0~9 の整数
    アクションA: 1~4の整数
    状態S': (S + A) % 10
    報酬R:
        +1: S' == 7
        -100: S' in (5, 9)
    """

    reward_type = 1

    def init_state(self):
        self.actions = [1, 2, 3, 4]
        self.state = 0

    def get_next_state_and_reward(self, state, action):
        next_state = (state + action) % 10
        reward = 0
        if self.reward_type == 1:
            if next_state == 7:
                reward = 1
            elif next_state in (5, 9):
                reward = -100
        return next_state, reward

class HumanPlayer(object):
    def action(self, state, last_reward):
        print "LastReward=%s, CurrentState: %s" % (last_reward, state)
        while True:
            action = raw_input("Enter 1~4: ")
            if int(action) in [1, 2, 3, 4]:
                break
        return int(action)

class QLearningPlayer(object):
    ALPHA = 0.1
    GAMMA = 0.99
    E_GREEDY = 0.01

    def __init__(self):
        self.actions = [1, 2, 3, 4]

        # Q tableは全て np.array でも良いんだけど、なんとなくHashにした方がわかりやすいので、最初はこう書いておく
        self.q_table = {}
        for s in range(10):  # 全状態は10個で 0~9
            self.q_table[s] = np.random.random([len(self.actions)])
        self.last_state = self.last_action = None
        self.training = True

    def action(self, state, last_reward):
        next_action = self.select_action(state)
        if self.last_state is not None:
            self.update_q_table(self.last_state, self.last_action, state, last_reward)
        self.last_state = state
        self.last_action = next_action
        return next_action

    def select_action(self, state):
        if self.training and random() < self.E_GREEDY:  # http://www.sist.ac.jp/~kanakubo/research/reinforcement_learning.html
            return choice(self.actions)
        else:
            return np.argmax(self.q_table[state]) + 1

    def update_q_table(self, last_state, last_action, cur_state, last_reward):
        if self.training:
            d = last_reward + np.max(self.q_table[cur_state]) * self.GAMMA - self.q_table[last_state][last_action-1]
            self.q_table[last_state][last_action-1] += self.ALPHA * d

if __name__ == '__main__':
    fp = file("result.txt", "w")
    player = QLearningPlayer()
    game = AddingGame(player)
    last_score = 0
    for g in game.play():
        if g.turn % 100 == 0:
            if not player.training:
                this_term_score = game.total_reward - last_score
                print "Turn %d: This 100 turn score: %s" % (g.turn, this_term_score)
                fp.write("%d\t%d\n" % (g.turn, this_term_score))
            last_score = game.total_reward
            player.training = not player.training
