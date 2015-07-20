#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
from random import random
import itertools

from chainer import FunctionSet
import chainer.functions as F

from game_common.base_system import AsciiGame, Screen
from game_common.debug_game import debug_game
from game_common.ascii_game_player_agent import agent_play
from game_common.agent_model import AgentModel
from game_common.agent_model import EmbedAgentModel


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
    POWER_MAX = 3

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

        ######## decide reward
        reward = 0.05

        # key penalty
        if action not in self.effective_actions():  # 余分なKeyを押したらペナルティとする(親切)
            reward -= 0.1

        if state.py < self.PY_MAX and screen[self.PY_MAX+1, state.px] == self.SPACE: # 穴の上では報酬がある
            reward += 0.1

        # game over?
        if state.py == self.PY_MAX and screen[self.PY_MAX+1, state.px] == self.SPACE:
            self.is_game_over = True
            reward = -1

        return state, reward

    # should be defined
    def effective_actions(self):
        ret = set()
        key_combinations = [(0, self.BUTTON_A), (0, self.KEY_LEFT, self.KEY_RIGHT)]
        for key_code_list in itertools.product(*key_combinations):
            ret.add(reduce(lambda t, x: t | x, key_code_list))
        return list(ret)

    # private methods
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

        if action & self.KEY_LEFT and state.px > 0:
            state.px -= 1
        if action & self.KEY_RIGHT and state.px < self.WIDTH-1:
            state.px += 1

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
                    state.power = self.POWER_MAX
                state.jumping_down = False

if __name__ == '__main__':
    def calc_output_size(screen_size, ksize, stride):
        return (screen_size - ksize) / stride + 1

    if os.environ.get("DEBUG_PLAY", None):
        print "Debug Mode"
        debug_game(JumpGame)
    elif os.environ.get("IN_TYPE", None) == 'id':
        print "EmbedID Mode"
        HISTORY_SIZE = 4
        PATTERN_SIZE = 100
        EMBED_OUT_SIZE = 4
        KSIZE = (8, 8*EMBED_OUT_SIZE)
        STRIDE = (4, 4*EMBED_OUT_SIZE)
        nw = calc_output_size(JumpGame.WIDTH*EMBED_OUT_SIZE, KSIZE[1], STRIDE[1])   # 9
        nh = calc_output_size(JumpGame.HEIGHT, KSIZE[0], STRIDE[0])  # 5
        chainer_model = FunctionSet(
            l1=F.Convolution2D(HISTORY_SIZE, PATTERN_SIZE, ksize=KSIZE, stride=STRIDE),
            l2=F.Linear(nw * nh * PATTERN_SIZE, 800),
            l3=F.Linear(800, 400),
            l4=F.Linear(400, 64),
        )
        model = EmbedAgentModel(model=chainer_model, model_name='JumpGameEmbedModel',
                                embed_out_size=EMBED_OUT_SIZE,
                                width=JumpGame.WIDTH, height=JumpGame.HEIGHT,
                                history_size=HISTORY_SIZE, out_size=64)
        model.activate_functions["l1"] = F.sigmoid
        model.activate_functions["l2"] = F.sigmoid
        model.activate_functions["l3"] = F.sigmoid
        agent_play(JumpGame, agent_model=model)
    else:
        print "Ver1 Mode"
        HISTORY_SIZE = 4
        PATTERN_SIZE = 100
        nw = calc_output_size(JumpGame.WIDTH, 8, 4)   # 9
        nh = calc_output_size(JumpGame.HEIGHT, 8, 4)  # 5
        chainer_model = FunctionSet(
            l1=F.Convolution2D(HISTORY_SIZE, PATTERN_SIZE, ksize=8, stride=4),
            l2=F.Linear(nw * nh * PATTERN_SIZE, 800),
            l3=F.Linear(800, 400),
            l4=F.Linear(400, 64),
        )
        model = AgentModel(model=chainer_model, model_name='JumpGame',
                           width=JumpGame.WIDTH, height=JumpGame.HEIGHT,
                           history_size=HISTORY_SIZE, out_size=64)
        agent_play(JumpGame, agent_model=model)
