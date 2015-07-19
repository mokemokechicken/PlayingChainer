#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
from random import random

from chainer import FunctionSet
import chainer.functions as F

from game_common.base_system import AsciiGame, Screen
from game_common.debug_game import debug_game
from game_common.ascii_game_player_agent import agent_play
from game_common.agent_model import AgentModel


class State(object):
    screen = None
    px = 5
    py = 20
    power = 3
    jumping_down = False

class JumpGame(AsciiGame):
    PLAYER = ord("P")
    BLOCK = ord("=")
    SPACE = ord(" ")
    space_rate = 0.2
    PY_MAX = 20
    POWER_MAX = 8

    # must be defined
    def prepare_game(self):
        self.state = State()
        self.state.screen = Screen(self.WIDTH, self.HEIGHT)
        self.init_course()
        self.draw_player()

    # must be defined
    def get_next_state_and_reward(self, state, action):
        self.move_player(state, action)
        # Scroll
        screen = state.screen
        screen.scroll_x(-1)
        screen[self.PY_MAX+1, self.WIDTH-1] = self.gen_block_or_space()
        self.draw_player()

        # game over?
        reward = 0.05
        if state.py == self.PY_MAX and screen[self.PY_MAX+1, state.px] == self.SPACE:
            self.is_game_over = True
            reward = -1

        return state, reward

    def init_course(self):
        self.state.screen.fill(self.SPACE)
        self.state.screen[self.PY_MAX+1] = [self.gen_block_or_space(0.05) for _ in range(self.WIDTH)]

    def gen_block_or_space(self, space_rate=None):
        space_rate = space_rate or self.space_rate
        return self.SPACE if random() < space_rate else self.BLOCK

    def draw_player(self, erase=False):
        self.state.screen[self.state.py, self.state.px] = self.SPACE if erase else self.PLAYER

    def move_player(self, state, action):
        # MOVE Player
        self.draw_player(erase=True)
        jump_key_pressed = action & self.BUTTON_A
        if jump_key_pressed and state.power > 0 and not state.jumping_down:
            state.power -= 1
            if state.power == 0:
                state.jumping_down = True
            state.py -= 1
        else:
            if state.py < self.PY_MAX:
                state.py += 1
                state.jumping_down = True
            else:
                if state.power < self.POWER_MAX:
                    state.power += 1
                state.jumping_down = False

if __name__ == '__main__':
    if os.environ.get("DEBUG", None):
        debug_game(JumpGame)
    else:
        chainer_model = FunctionSet(
            l1=F.Linear(40*24, 800),
            l2=F.Linear(800, 500),
            l3=F.Linear(500, 300),
            l4=F.Linear(300, 64),
        )
        model = AgentModel(chainer_model, 'JumpGame', in_size=JumpGame.WIDTH*JumpGame.HEIGHT, out_size=64)
        agent_play(JumpGame, agent_model=model)
