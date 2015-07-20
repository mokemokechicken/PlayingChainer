#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'


import os
import copy
from random import random, randint
import itertools

from chainer import FunctionSet
import chainer.functions as F

from game_common.base_system import AsciiGame, Screen
from game_common.debug_game import debug_game
from game_common.ascii_game_player_agent import agent_play
from game_common.agent_model import EmbedAgentModel

class Pos(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return isinstance(other, Pos) and self.x == other.x and self.y == other.y

class Treasure(object):
    def __init__(self, pos, time):
        self.pos = pos
        self.time = time

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

class State(object):
    screen = None

    def __init__(self):
        self.player_pos = Pos(20, 12)
        self.enemy_list = [EnemyX(Pos(3, 3), 4), EnemyY(Pos(37, 21), 3)]
        self.treasure_list = []
        self.treasure_pop_timer = 0


class TreasureGame(AsciiGame):
    SPACE = ord(" ")
    PLAYER = ord("A")
    TREASURE = ord("$")

    LIFE_OF_TREASURE = 40
    TREASURE_POP_SPAN = 8

    # must be defined
    def prepare_game(self):
        self.state = State()
        self.state.screen = Screen(self.WIDTH, self.HEIGHT)
        self.init_course()
        self.draw()

    # must be defined
    def get_next_state_and_reward(self, state, action):
        self.move_player(state, action)
        self.move_enemies(state)
        self.count_down_treasures()

        state.treasure_pop_timer += 1
        if state.treasure_pop_timer == self.TREASURE_POP_SPAN:
            state.treasure_pop_timer = 0
            self.pop_treasure()

        # decide reward
        reward = 0

        # key penalty -0.1
        if action not in self.effective_actions():  # 余分なKeyを押したらペナルティとする(親切)
            reward -= 0.1

        # got treasure? +0.5
        pos = state.player_pos
        for t in copy.copy(state.treasure_list):
            if t.pos == pos:
                reward += 0.5
                state.treasure_list.remove(t)

        # game over?
        for e in self.state.enemy_list:
            if e.pos == pos:
                self.is_game_over = True
                reward = -1
                break

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
        self.state.screen.fill(self.SPACE)
        for _ in range(5):
            self.pop_treasure()

    def draw(self):
        screen = self.state.screen
        screen.fill(self.SPACE)
        for t in self.state.treasure_list:
            screen[t.pos.y, t.pos.x] = self.TREASURE
        screen[self.state.player_pos.y, self.state.player_pos.x] = self.PLAYER
        for e in self.state.enemy_list:
            screen[e.pos.y, e.pos.x] = e.CHAR

    def pop_treasure(self):
        while True:
            pos = Pos(randint(0, self.WIDTH-1), randint(0, self.HEIGHT-1))
            if self.state.screen[pos.y, pos.x] == self.SPACE:
                t = Treasure(pos, self.LIFE_OF_TREASURE)
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

    def count_down_treasures(self):
        for t in copy.copy(self.state.treasure_list):
            t.time -= 1
            if t.time == 0:
                self.state.treasure_list.remove(t)


if __name__ == '__main__':
    ThisGame = TreasureGame

    def calc_output_size(screen_size, ksize, stride):
        return (screen_size - ksize) / stride + 1

    if os.environ.get("DEBUG_PLAY", None):
        print "Debug Mode"
        debug_game(ThisGame)
    else:
        print "EmbedID Mode"
        HISTORY_SIZE = 4
        PATTERN_SIZE = 100
        EMBED_OUT_SIZE = 4
        KSIZE = (8, 8*EMBED_OUT_SIZE)
        STRIDE = (4, 4*EMBED_OUT_SIZE)
        nw = calc_output_size(ThisGame.WIDTH*EMBED_OUT_SIZE, KSIZE[1], STRIDE[1])   # 9
        nh = calc_output_size(ThisGame.HEIGHT, KSIZE[0], STRIDE[0])  # 5
        chainer_model = FunctionSet(
            l1=F.Convolution2D(HISTORY_SIZE, PATTERN_SIZE, ksize=KSIZE, stride=STRIDE),
            l2=F.Linear(nw * nh * PATTERN_SIZE, 800),
            l3=F.Linear(800, 64),
            # l3=F.Linear(800, 400),
            # l4=F.Linear(400, 64),
        )
        model = EmbedAgentModel(model=chainer_model, model_name='TreasureGameEmbedModel',
                                embed_out_size=EMBED_OUT_SIZE,
                                width=ThisGame.WIDTH, height=ThisGame.HEIGHT,
                                history_size=HISTORY_SIZE, out_size=64)
        model.activate_functions["l1"] = F.relu
        model.activate_functions["l2"] = F.relu
        agent_play(ThisGame, agent_model=model)
