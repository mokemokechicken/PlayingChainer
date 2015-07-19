# coding: utf-8
__author__ = 'k_morishita'

import os
from random import random, randint, choice

import numpy as np
from chainer import Variable, optimizers

from game_repository import GameRepository
from replay_server import ReplayServer


class AsciiGamePlayerAgent(object):
    ALPHA = 0.1
    GAMMA = 0.99
    E_GREEDY = 0.3

    optimizer = None

    last_action = None
    last_q_list = None
    training = True
    use_greedy = True

    def __init__(self, agent_model, repo=None):
        """

        :type agent_model: AgentModel
        """
        self.agent_model = agent_model
        self.actions = range(self.agent_model.out_size)
        self.repo = repo or GameRepository()
        self.load_model_parameters()
        self.effective_action_index_list = range(len(self.actions))

    def load_model_parameters(self):
        self.repo.load_model_params(self.agent_model)
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.agent_model.function_set.collect_parameters())

    def ready(self):
        self.last_action = None
        self.last_q_list = None

    def action(self, state, last_reward):
        if self.last_action is not None and self.training:
            self.update_q_table(self.last_action, state, last_reward)
        next_action = self.select_action(state)
        self.last_action = next_action
        return self.actions[next_action]

    def convert_state_to_input(self, state):
        in_data = ((state.screen.data - 32) / 96.0).astype('float32').reshape(self.agent_model.in_size)
        return Variable(np.array([in_data]))

    def forward(self, state):
        x = self.convert_state_to_input(state)
        return self.agent_model.forward(x)

    def select_action(self, state):
        self.last_q_list = self.forward(state)
        if self.use_greedy and random() < self.E_GREEDY:
            return choice(self.effective_action_index_list)
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
        self.agent_model.on_learn(times=len(tt))

    # Game Life cycle
    def on_game_start(self, game):
        self.ready()

    def on_game_over(self, game):
        # Learn last bad reward
        self.action(game.state, game.last_reward)
        if self.training:
            self.repo.save_model_params(self.agent_model)

    def on_update(self, game):
        pass

def agent_play(game_class, agent_model):
    player = AsciiGamePlayerAgent(agent_model)
    replay_server = ReplayServer(int(os.environ.get("GAME_SERVER_PORT", 7000)))

    game = game_class(player)
    game.add_observer(replay_server)
    game.add_observer(player)

    replay_server.run_as_background()

    player.effective_action_index_list = game.effective_actions()

    while True:
        replay_server.info = ["e-Greedy=%s" % player.use_greedy, agent_model.meta_str()]
        game.play()
        if game.play_id % 10 == 0:
            player.use_greedy = not player.use_greedy
