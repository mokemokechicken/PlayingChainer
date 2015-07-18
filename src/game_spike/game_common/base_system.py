#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import numpy as np


class Game(object):
    observers = []

    state = None
    turn = None
    last_reward = None
    total_reward = None
    is_game_over = None
    last_action = None
    high_score = 0

    def __init__(self, player):
        self.player = player
        self.play_id = 0

    def game_start(self):
        self.state = None
        self.turn = 0
        self.last_reward = 0
        self.total_reward = 0
        self.is_game_over = False
        self.play_id += 1
        self.prepare_game()
        self.notify_game_start()

    def player_action(self):
        action = self.player.action(self.state, self.last_reward)
        if not self.is_valid_action(action):
            action = None
        self.state, self.last_reward = self.get_next_state_and_reward(self.state, action)
        self.last_action = action

    def game_over(self):
        self.notify_game_over()

    def play(self):
        self.game_start()
        while not self.is_game_over:
            self.turn += 1
            self.player_action()
            self.total_reward += self.last_reward
            self.notify_update()
        self.game_over()
        if self.high_score < self.total_reward:
            self.high_score = self.total_reward

    def turn_info(self):
        return {
            "turn": self.turn,
            "last_reward": self.last_reward,
            "total_reward": self.total_reward,
            "last_action": self.last_action,
        }

    def meta_info(self):
        return {
            "play_id": self.play_id,
            "high_score": self.high_score,
        }

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


class Screen(object):
    data = []

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.init_screen()

    def init_screen(self):
        self.data = np.zeros([self.height, self.width], dtype=np.int32)

    def fill(self, ch):
        self.data.fill(ch)

    def as_float32(self):
        return self.data.astype(np.float32)

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def scroll_x(self, shift):
        self.data = np.roll(self.data, shift, axis=1)

    def scroll_y(self, shift):
        self.data = np.roll(self.data, shift, axis=0)



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

    def meta_info(self):
        info = super(AsciiGame, self).meta_info()
        info["keymap"] = {
            "UP": self.KEY_UP,
            "DOWN": self.KEY_DOWN,
            "RIGHT": self.KEY_RIGHT,
            "LEFT": self.KEY_LEFT,
            "A": self.BUTTON_A,
            "B": self.BUTTON_B,
        }
        return info