# coding: utf-8
__author__ = 'k_morishita'

import os
from random import random, randint, choice
import math
import numpy as np
from chainer import Variable, optimizers

from game_repository import GameRepository
from replay_server import ReplayServer

class LossHistory(object):
    pointer = 0
    history_ready = False

    def __init__(self, size):
        self.loss_history = None
        self.max_loss_in_a_game = 0
        self.loss_history_size = size
        self.reset()

    def ready(self):
        self.max_loss_in_a_game = 0

    def reset(self):
        self.pointer = 0
        self.loss_history = np.zeros([self.loss_history_size], dtype=np.float32)
        self.history_ready = False

    def add_loss(self, loss):
        loss_z = (loss - np.average(self.loss_history)) / np.var(self.loss_history)
        self.loss_history[self.pointer] = loss
        self.pointer = (self.pointer + 1) % self.loss_history_size
        if self.pointer == 0:
            self.history_ready = True
        self.max_loss_in_a_game = max(self.max_loss_in_a_game, loss)
        return loss_z

class QuitGameException(Exception):
    pass

class AsciiGamePlayerAgent(object):
    ALPHA = 0.1
    GAMMA = 0.99
    E_GREEDY = 0.3

    optimizer = None
    history_data = None

    last_state = None
    last_action = None
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
        self.loss_history = LossHistory(100)

    def load_model_parameters(self):
        self.repo.load_model_params(self.agent_model)
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.agent_model.function_set.collect_parameters())

    def ready(self):
        self.last_action = None
        self.loss_history.ready()
        self.history_data = np.zeros([self.agent_model.history_size, self.agent_model.height, self.agent_model.width],
                                     dtype=np.float32)

    def action(self, state, last_reward):
        if self.last_action is not None and self.training:
            self.update_q_table(self.last_state, self.last_action, state, last_reward)
        next_action = self.select_action(state)
        self.last_action = next_action
        self.last_state = state.deepcopy()
        return self.actions[next_action]

    def convert_state_to_input(self, state):
        in_data = self.agent_model.convert_state_to_input(state)
        self.history_data = np.roll(self.history_data, -1, axis=0)
        self.history_data[self.agent_model.history_size-1] = in_data
        return Variable(self.history_data.reshape((1, self.agent_model.history_size,
                                                   self.agent_model.height, self.agent_model.width)))

    def forward(self, state, train=True):
        x = self.convert_state_to_input(state)
        return self.agent_model.forward(x, train=train)

    def select_action(self, state):
        q_list = self.forward(state, train=False)
        if self.use_greedy and random() < self.E_GREEDY:
            return choice(self.effective_action_index_list)
        else:
            return np.argmax(q_list.data)

    def update_q_table(self, last_state, last_action, cur_state, last_reward):
        for loop_num in range(100):
            loss_value = self.do_update_q_table(last_state, last_action, cur_state, last_reward)
            loss_z = self.loss_history.add_loss(loss_value)
            if is_debug():
                print "loss=%s\tZ=%s\tmax_loss=%s\tLOOP=%s" % \
                      (round(loss_value, 6), round(loss_z, 2),
                       self.loss_history.max_loss_in_a_game, loop_num)
            if loss_z < 4 or not self.loss_history.history_ready:
                break
            if math.isnan(loss_value):
                self.loss_history.reset()
                raise QuitGameException("loss_value become Nan!")

    def do_update_q_table(self, last_state, last_action, cur_state, last_reward):
        target_val = last_reward + self.GAMMA * np.max(self.forward(cur_state, train=False).data)

        self.optimizer.zero_grads()
        last_q_list = self.forward(last_state)
        tt = np.copy(last_q_list.data)
        tt[0][last_action] = target_val
        target = Variable(tt)
        loss = 0.5 * (target - last_q_list) ** 2
        loss_value = loss.data[0][last_action]
        loss.grad = np.array([[self.ALPHA]], dtype=np.float32)
        loss.backward()
        self.optimizer.update()
        self.agent_model.on_learn(times=len(tt))
        return loss_value

    # Game Life cycle
    def on_game_start(self, game):
        self.ready()

    def on_game_over(self, game):
        # Learn last bad reward
        self.action(game.state, game.last_reward)
        if self.training:
            self.repo.save_model_params(self.agent_model)
        if is_debug():
            print "max_loss_in_this_game: %s" % self.loss_history.max_loss_in_a_game

    def on_update(self, game):
        pass

def is_debug():
    return os.environ.get("DEBUG", None) is not None

def agent_play(game_class, agent_player):
    replay_server = ReplayServer(int(os.environ.get("GAME_SERVER_PORT", 7000)))

    game = game_class(agent_player)
    game.add_observer(replay_server)
    game.add_observer(agent_player)

    replay_server.run_as_background()

    agent_player.effective_action_index_list = game.effective_actions()

    while True:
        replay_server.info = ["e-Greedy=%s" % agent_player.use_greedy] + agent_player.agent_model.info_list()
        try:
            game.play()
            if game.play_id % 10 == 0:
                agent_player.use_greedy = not agent_player.use_greedy
        except QuitGameException as e:
            agent_player.load_model_parameters()
            print e
