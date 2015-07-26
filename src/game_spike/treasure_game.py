#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
import copy

import random
import itertools

from game_common.base_system import AsciiGame, Screen, StateBase
from game_common.debug_game import debug_game
from game_common.runner import run_pattern1, run_pattern2


class Pos(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Pos) and self.x == other.x and self.y == other.y

class Treasure(object):
    def __init__(self, pos):
        self.pos = pos

class Enemy(object):
    def __init__(self, pos, counter_max):
        self.pos = pos
        self.counter_max = counter_max
        self.counter = counter_max

    def move(self, state):
        self.counter -= 1
        if self.counter == 0:
            self.counter = self.counter_max
            self.do_move(state)

def sign(x):
    if x == 0:
        return 0
    elif x < 0:
        return -1
    else:
        return 1

class EnemyX(Enemy):
    """遅いけどPlayer に向かって一直線"""
    CHAR = ord("X")

    def do_move(self, state):
        self.pos.x += sign(state.player_pos.x - self.pos.x)
        self.pos.y += sign(state.player_pos.y - self.pos.y)

class EnemyY(Enemy):
    """ちょっと速いけど、縦か横にしか動けない"""
    CHAR = ord("Y")

    def do_move(self, state):
        dx = state.player_pos.x - self.pos.x
        dy = state.player_pos.y - self.pos.y
        if abs(dx) <= abs(dy):
            self.pos.y += sign(state.player_pos.y - self.pos.y)
        else:
            self.pos.x += sign(state.player_pos.x - self.pos.x)

class State(StateBase):
    def __init__(self):
        self.player_pos = Pos(7, 4)
        self.enemy_list = [EnemyY(Pos(13, 9), 2)]
        self.treasure_list = []
        self.treasure_pop_timer = 0


class TreasureGame(AsciiGame):
    SPACE = ord(" ")
    PLAYER = ord("A")
    TREASURE = ord("$")

    NUM_TREASURES = 20
    MAX_TURN = 800
    stage = 0

    # must be defined
    def prepare_game(self):
        self.state = State()
        self.state.screen = Screen(self.WIDTH, self.HEIGHT)
        self.init_course()
        self.draw()

    def add_enemy(self):
        self.state.enemy_list.append(EnemyX(Pos(0, 0), 3))

    # must be defined
    def get_next_state_and_reward(self, state, action):
        self.move_player(state, action)
        self.move_enemies(state)

        # decide reward
        reward = 0.01

        # key penalty -0.5
        # if action not in self.effective_actions():  # 余分なKeyを押したらペナルティとする(親切 余計な？)
        #     reward -= 0.5

        # got treasure? +0.1
        pos = state.player_pos
        for t in copy.copy(state.treasure_list):
            if t.pos == pos:
                reward += 0.2
                state.treasure_list.remove(t)
                if len(state.treasure_list) == 0:
                    self.pop_treasures()
                    self.add_enemy()

        # game over?
        for e in self.state.enemy_list:
            if e.pos == pos:
                self.is_game_over = True
                reward = -1
                break

        if self.turn == self.MAX_TURN:
            self.is_game_over = True

        self.draw()
        return state, reward

    # should be defined
    def effective_actions(self):
        ret = set()
        key_combinations = [(0, self.KEY_UP, self.KEY_DOWN), (0, self.KEY_LEFT, self.KEY_RIGHT)]
        for key_code_list in itertools.product(*key_combinations):
            ret.add(reduce(lambda t, x: t | x, key_code_list))
        return list(ret)

    # private methods
    def init_course(self):
        self.stage = 0
        self.state.screen.fill(self.SPACE)
        self.pop_treasures()

    def draw(self):
        screen = self.state.screen
        screen.fill(self.SPACE)
        for t in self.state.treasure_list:
            screen[t.pos.y, t.pos.x] = self.TREASURE
        screen[self.state.player_pos.y, self.state.player_pos.x] = self.PLAYER
        for e in self.state.enemy_list:
            screen[e.pos.y, e.pos.x] = e.CHAR

    def pop_treasures(self):
        self.stage += 1
        random.seed(self.stage * 1000)
        for _ in range(self.NUM_TREASURES):
            self.pop_treasure()

    def pop_treasure(self):
        while True:
            pos = Pos(random.randint(0, self.WIDTH-1), random.randint(0, self.HEIGHT-1))
            if self.state.screen[pos.y, pos.x] == self.SPACE:
                t = Treasure(pos)
                self.state.treasure_list.append(t)
                break

    def move_player(self, state, action):
        pos = state.player_pos
        if action & self.KEY_LEFT and pos.x > 0:
            pos.x -= 1
        if action & self.KEY_RIGHT and pos.x < self.WIDTH-1:
            pos.x += 1
        if action & self.KEY_UP and pos.y > 0:
            pos.y -= 1
        if action & self.KEY_DOWN and pos.y < self.HEIGHT-1:
            pos.y += 1

    def move_enemies(self, state):
        for e in state.enemy_list:
            e.move(state)

if __name__ == '__main__':

    if os.environ.get("DEBUG_PLAY", None):
        print "Debug Mode"
        debug_game(TreasureGame)
    elif os.environ.get("MODEL") == '2':
        print "Model2"
        run_pattern2(TreasureGame, 'TreasureGameModeL2')
    else:
        print "EmbedID Mode"
        run_pattern1(TreasureGame, 'TreasureGameEmbedModel')
