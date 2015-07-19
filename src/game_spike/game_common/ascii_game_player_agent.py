# coding: utf-8
__author__ = 'k_morishita'

import os
from random import random, randint
import datetime

import numpy as np
from chainer import FunctionSet, Variable, optimizers
import chainer.functions as F

from game_repository import GameRepository
from replay_server import ReplayServer


class AsciiGamePlayerAgent(object):
    ALPHA = 0.1
    GAMMA = 0.99
    E_GREEDY = 0.3

    def __init__(self, model=None, repo=None, model_name=None):
        """

        :type model: FunctionSet
        """
        self.actions = range(64)
        self.model_name = model_name or ('created_%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.model = model or FunctionSet(
            l1=F.Linear(40*24, 800),
            l2=F.Linear(800, 500),
            l3=F.Linear(500, 300),
            l4=F.Linear(300, len(self.actions)),
        )
        self.in_size = len(self.model.l1.W[0])
        self.repo = repo or GameRepository()

        self.repo.load_model_params(self.model, self.model_name)
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model.collect_parameters())
        self.last_action = None
        self.last_q_list = None
        self.training = True

    def action(self, state, last_reward):
        if self.last_action is not None and self.training:
            self.update_q_table(self.last_action, state, last_reward)
        next_action = self.select_action(state)
        self.last_action = next_action
        return self.actions[next_action]

    def convert_state_to_input(self, state):
        in_data = ((state.screen.data - 32) / 96.0).astype('float32').reshape(self.in_size)
        return Variable(np.array([in_data]))

    def forward(self, state):
        x = self.convert_state_to_input(state)
        y = None
        for i in range(1, 1000):  # 1000 は適当な数
            if hasattr(self.model, "l%d" % i):
                x = getattr(self.model, "l%d" % i)(x)
            else:
                y = x
                break
        return y

    def select_action(self, state):
        self.last_q_list = self.forward(state)
        if self.training and random() < self.E_GREEDY:
            return randint(0, len(self.actions)-1)
        else:
            return np.argmax(self.last_q_list.data)

    def update_q_table(self, last_action, cur_state, last_reward):
        target_val = last_reward + self.GAMMA * np.max(self.forward(cur_state).data)
        self.optimizer.zero_grads()
        # 結構無理やりLossを計算・・・ この辺の実装は自信がない
        tt = np.copy(self.last_q_list.data)
        tt[0][last_action] = target_val
        target = Variable(tt)
        loss = 0.5 * (target - self.last_q_list) ** 2
        loss.grad = np.array([[self.ALPHA]], dtype=np.float32)
        loss.backward()
        self.optimizer.update()

def agent_play(game_class, model=None):
    player = AsciiGamePlayerAgent(model=model, model_name=game_class.__name__)
    replay_server = ReplayServer(int(os.environ.get("GAME_SERVER_PORT", 7000)))

    game = game_class(player)
    game.add_observer(replay_server)

    replay_server.run_as_background()

    while True:
        game.play()

