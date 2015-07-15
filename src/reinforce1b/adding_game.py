#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

"""
問題： 以下の様なGameを強化学習+NeuralNetで学べるか
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
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

from random import random, choice, randint


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
                reward = 0.01
            elif next_state in (5, 9):
                reward = -1.0
        return next_state, reward

class HumanPlayer(object):
    def action(self, state, last_reward):
        print "LastReward=%s, CurrentState: %s" % (last_reward, state)
        while True:
            action = raw_input("Enter 1~4: ")
            if int(action) in [1, 2, 3, 4]:
                break
        return int(action)

class NNQLearningPlayer(object):
    ALPHA = 0.05
    GAMMA = 0.99
    E_GREEDY = 0.1

    def __init__(self):
        self.actions = [1, 2, 3, 4]
        self.model = FunctionSet(
            l1=F.Linear(1, 20),
            l2=F.Linear(20, 10),
            l3=F.Linear(10, 4),
        )
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model.collect_parameters())
        self.last_state = self.last_action = None
        self.last_q_list = None
        self.training = True

    def action(self, state_, last_reward):
        state = state_ / 9.0
        if self.last_state is not None and self.training:
            self.update_q_table(self.last_state, self.last_action, state, last_reward)
        next_action = self.select_action(state)
        self.last_state = state
        self.last_action = next_action
        return self.actions[next_action]

    def forward(self, state, volatile=False):
        x = Variable(np.array([[state]], dtype=np.float32), volatile=volatile)
        for i in range(1, 1000):  # 1000 は適当な数
            if hasattr(self.model, "l%d" % i):
                # if i > 1:
                #     x = F.relu(x)
                x = getattr(self.model, "l%d" % i)(x)
            else:
                y = x
                break
        return y

    def select_action(self, state):
        self.last_q_list = self.forward(state)
        if self.training and random() < self.E_GREEDY:  # http://www.sist.ac.jp/~kanakubo/research/reinforcement_learning.html
            return randint(0, len(self.actions)-1)
        else:
            # actions = self.forward(state, volatile=True)
            return np.argmax(self.last_q_list.data)

    def update_q_table(self, last_state, last_action, cur_state, last_reward):
        target_val = last_reward + self.GAMMA * np.max(self.forward(cur_state, volatile=True).data)
        self.optimizer.zero_grads()
        q_last = self.last_q_list
        tt = np.copy(q_last.data)
        tt[0][last_action] = target_val
        target = Variable(tt)
        loss = 0.5 * (target - q_last) ** 2
        loss.grad = np.array([[self.ALPHA]], dtype=np.float32)
        loss.backward()
        self.optimizer.update()


if __name__ == '__main__':
    fp = file("result.txt", "w")
    player = NNQLearningPlayer()
    game = AddingGame(player)
    last_score = 0
    for g in game.play():
        if g.turn % 100 == 0:
            if not player.training:
                this_term_score = int(round((game.total_reward - last_score)*100))
                print "Turn %d: This 100 turn score: %s" % (g.turn, this_term_score)
                fp.write("%d\t%d\n" % (g.turn, this_term_score))
            last_score = game.total_reward
            player.training = not player.training
