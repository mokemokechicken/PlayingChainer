#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
import random
import itertools

from game_common.base_system import AsciiGame, Screen, StateBase
from game_common.debug_game import debug_game
from game_common.runner import run_pattern1


class State(StateBase):
    px = 5
    py = 7
    power = 3
    jumping_down = False

class JumpGame(AsciiGame):
    PLAYER = ord("P")
    BLOCK = ord("=")
    SPACE = ord(" ")
    space_rate = 0.2
    PY_MAX = 8
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

        # if state.py < self.PY_MAX and screen[self.PY_MAX+1, state.px] == self.SPACE:  # 穴の上では報酬がある
        #     reward += 0.1

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


if __name__ == '__main__':
    if os.environ.get("DEBUG_PLAY", None):
        print "Debug Mode"
        debug_game(JumpGame)
    else:
        print "EmbedID Mode"
        run_pattern1(JumpGame, 'JumpGame')

