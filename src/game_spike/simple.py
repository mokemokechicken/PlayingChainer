#!/usr/bin/env python
# coding: utf-8

from random import random

__author__ = 'k_morishita'

import numpy as np

class Game(object):
    observers = []

    state = None
    turn = None
    last_reward = None
    total_reward = None
    is_game_over = None

    def __init__(self, player):
        self.player = player

    def game_start(self):
        self.state = None
        self.turn = 0
        self.last_reward = 0
        self.total_reward = 0
        self.is_game_over = False
        self.prepare_game()
        self.notify_game_start()

    def player_action(self):
        action = self.player.action(self.state, self.last_reward)
        if not self.is_valid_action(action):
            raise Exception("Invalid Action: '%s'" % action)
        self.state, self.last_reward = self.get_next_state_and_reward(self.state, action)
        self.notify_update()

    def game_over(self):
        self.notify_game_over()

    def play(self):
        self.game_start()
        while not self.is_game_over:
            self.player_action()
            self.turn += 1
            self.total_reward += self.last_reward
            yield (self)
        self.game_over()

    def prepare_game(self):
        raise NotImplemented()

    def is_valid_action(self, action):
        return True

    def get_next_state_and_reward(self, state, action):
        raise NotImplemented()

    ###################
    # Observer
    ###################
    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_game_start(self):
        for o in self.observers:
            o.on_game_start(self)

    def notify_game_over(self):
        for o in self.observers:
            o.on_game_over(self)

    def notify_update(self):
        for o in self.observers:
            o.on_update(self)

class AsciiGame(Game):
    WIDTH = 40
    HEIGHT = 24
    KEY_UP = 1 << 0    # 1
    KEY_DOWN = 1 << 1  # 2
    KEY_RIGHT = 1 << 2
    KEY_LEFT = 1 << 3
    BUTTON_A = 1 << 4
    BUTTON_B = 1 << 5

    def is_valid_action(self, action):
        return 0 <= int(action) < 64

class Screen(object):
    data = []

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def init_screen(self):
        self.data = np.zeros([self.height, self.width], dtype=np.int32)

    def as_float32(self):
        return self.data.astype(np.float32)

    def __setitem__(self, key, value):
        self.data[key] = value

    def scroll_x(self, shift):
        np.roll(self.data, shift, axis=1)

    def scroll_y(self, shift):
        np.roll(self.data, shift, axis=0)


# ####### Game Dependent

class State(object):
    screen = Screen()
    px = None
    py = None

class JumpGame(AsciiGame):
    PLAYER = ord("P")
    BLOCK = ord("#")
    SPACE = ord(" ")
    space_rate = 0.2

    def prepare_game(self):
        self.state = State()
        self.state.px = 5
        self.state.py = 20
        self.init_course()
        self.draw_player()

    def init_course(self):
        self.state.screen[21] = [self.SPACE if random() < self.space_rate else self.BLOCK for _ in range(self.WIDTH)]

    def draw_player(self, erase=False):
        self.state.screen[self.state.py, self.state.px] = self.SPACE if erase else self.PLAYER

    def get_next_state_and_reward(self, state, action):
        screen = state.screen
        self.draw_player(erase=True)
        self.KEY_RIGHT


class AsciiGameScreenObserver(object):
    def on_game_start(self, game):
        pass

    def on_game_over(self, game):
        pass

    def on_update(self, game):
        pass

