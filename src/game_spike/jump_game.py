#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
import random
import itertools

from chainer import FunctionSet
import chainer.functions as F

from game_common.base_system import AsciiGame, Screen, StateBase
from game_common.debug_game import debug_game
from game_common.ascii_game_player_agent import agent_play, AsciiGamePlayerAgent
from game_common.agent_model import AgentModel
from game_common.agent_model import EmbedAgentModel


class State(StateBase):
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
        random.seed(self.turn+1)
        ret = self.SPACE if random.random() < space_rate else self.BLOCK
        random.seed(0)
        return ret

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


def ptn1(ThisGame, model_name):
    HISTORY_SIZE = 4
    PATTERN_SIZE1 = 20
    EMBED_OUT_SIZE = 4
    KSIZE1 = (3, 3*EMBED_OUT_SIZE)
    STRIDE1 = (1, 1*EMBED_OUT_SIZE)
    nw1 = calc_output_size(ThisGame.WIDTH*EMBED_OUT_SIZE, KSIZE1[1], STRIDE1[1])  # 38
    nh1 = calc_output_size(ThisGame.HEIGHT, KSIZE1[0], STRIDE1[0])                # 22

    PATTERN_SIZE2 = 100
    KSIZE2  = (4, 5)
    STRIDE2 = (3, 3)
    nw2 = calc_output_size(nw1, KSIZE2[1], STRIDE2[1])  # 11
    nh2 = calc_output_size(nh1, KSIZE2[0], STRIDE2[0])  # 6
    chainer_model = FunctionSet(
        l1=F.Convolution2D(HISTORY_SIZE, PATTERN_SIZE1, ksize=KSIZE1, stride=STRIDE1),
        l2=F.Convolution2D(PATTERN_SIZE1, PATTERN_SIZE2, ksize=KSIZE2, stride=STRIDE2),
        l3=F.Linear(nw2 * nh2 * PATTERN_SIZE2, 1000),
        l4=F.Linear(1000, 64),
    )
    model = EmbedAgentModel(model=chainer_model, model_name=model_name,
                            embed_out_size=EMBED_OUT_SIZE,
                            width=ThisGame.WIDTH, height=ThisGame.HEIGHT,
                            history_size=HISTORY_SIZE, out_size=64)

    def relu_with_drop_ratio(ratio):
        def f(x, train=True):
            return F.dropout(F.relu(x), train=train, ratio=ratio)
        return f

    def drop_ratio(ratio):
        def f(x, train=True):
            return F.dropout(x, train=train, ratio=ratio)
        return f

    model.activate_functions["l1"] = relu_with_drop_ratio(0.2)
    model.activate_functions["l2"] = relu_with_drop_ratio(0.4)
    model.activate_functions["l3"] = relu_with_drop_ratio(0.5)
    model.activate_functions["l4"] = drop_ratio(0.7)
    player = AsciiGamePlayerAgent(model)
    player.ALPHA = 0.01
    agent_play(ThisGame, player)


if __name__ == '__main__':
    def calc_output_size(screen_size, ksize, stride):
        return (screen_size - ksize) / stride + 1

    if os.environ.get("DEBUG_PLAY", None):
        print "Debug Mode"
        debug_game(JumpGame)
    else:
        print "EmbedID Mode"
        ptn1(JumpGame, 'JumpGame')

