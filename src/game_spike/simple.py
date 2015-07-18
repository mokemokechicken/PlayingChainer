#!/usr/bin/env python
# coding: utf-8
import curses

from random import random
import time

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


# ####### Game Dependent

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

    def prepare_game(self):
        self.state = State()
        self.state.screen = Screen(self.WIDTH, self.HEIGHT)
        self.init_course()
        self.draw_player()

    def init_course(self):
        self.state.screen.fill(self.SPACE)
        self.state.screen[self.PY_MAX+1] = [self.gen_block_or_space(0.05) for _ in range(self.WIDTH)]

    def gen_block_or_space(self, space_rate=None):
        space_rate = space_rate or self.space_rate
        return self.SPACE if random() < space_rate else self.BLOCK

    def draw_player(self, erase=False):
        self.state.screen[self.state.py, self.state.px] = self.SPACE if erase else self.PLAYER

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

    def move_player(self, state, action):
        # MOVE Player
        self.draw_player(erase=True)
        # action > 0 means ANY KEY
        if action > 0 and state.power > 0 and not state.jumping_down:
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


class DebugHumanPlayer(object):
    W_LEFT = 1
    W_TOP = 1
    main_window = None
    info_window = None
    keymap = {}
    MSEC_PER_TURN = 100
    SEC_PER_TURN = MSEC_PER_TURN / 1000.0

    INFO_WIDTH = 40

    def __init__(self, stdscr):
        self.stdscr = stdscr

    def setup_debug_screen(self, game):
        """

        :type game: AsciiGame
        """
        curses.curs_set(0)
        border_win = curses.newwin(game.HEIGHT+2, game.WIDTH+2, self.W_TOP, self.W_LEFT)  # h, w, y, x
        border_win.box()
        self.stdscr.refresh()
        border_win.refresh()
        self.main_window = curses.newwin(game.HEIGHT, game.WIDTH, self.W_TOP+1, self.W_LEFT+1)
        self.main_window.refresh()
        self.main_window.timeout(self.MSEC_PER_TURN - 5)

        self.info_window = curses.newwin(game.HEIGHT, self.INFO_WIDTH, self.W_TOP+1, self.W_LEFT+game.WIDTH + 2)

        self.keymap[ord("j")] = game.KEY_LEFT
        self.keymap[ord("l")] = game.KEY_RIGHT
        self.keymap[ord("i")] = game.KEY_UP
        self.keymap[ord("k")] = game.KEY_DOWN
        self.keymap[ord("m")] = game.KEY_DOWN
        self.keymap[ord("x")] = game.BUTTON_A
        self.keymap[ord(" ")] = game.BUTTON_A
        self.keymap[ord("z")] = game.BUTTON_B

    def action(self, state, last_reward):
        t1 = time.time()
        c = self.main_window.getch()
        t2 = time.time() - t1
        if t2 < self.SEC_PER_TURN:
            time.sleep(self.SEC_PER_TURN - t2)
        return self.keymap.get(c, 0)

    def on_game_start(self, game):
        pass

    def on_game_over(self, game):
        curses.flash()
        time.sleep(3)

    def on_update(self, game):
        """

        :type game: JumpGame
        """
        screen = game.state.screen
        for y in range(game.HEIGHT):
            line = "".join([chr(ch) for ch in screen.data[y]])
            ignore_error_add_str(self.main_window, y, 0, line)
        self.main_window.refresh()

        self.info_window.clear()
        self.info_window.addstr(1 , 2, "Total Score: %s" % game.total_reward)
        self.info_window.addstr(2 , 2, "This Reward: %s" % game.last_reward)
        self.info_window.addstr(3 , 2, "=== Action ===")
        self.info_window.addstr(4 , 2, "Left : %d" % (game.KEY_LEFT  & game.last_action > 0))
        self.info_window.addstr(5 , 2, "Right: %d" % (game.KEY_RIGHT & game.last_action > 0))
        self.info_window.addstr(6 , 2, "Up   : %d" % (game.KEY_UP    & game.last_action > 0))
        self.info_window.addstr(7 , 2, "Down : %d" % (game.KEY_DOWN  & game.last_action > 0))
        self.info_window.addstr(8 , 2, "A    : %d" % (game.BUTTON_A  & game.last_action > 0))
        self.info_window.addstr(9 , 2, "B    : %d" % (game.BUTTON_B  & game.last_action > 0))
        self.info_window.refresh()

def ignore_error_add_str(win, y, x, s):
    """一番右下に書き込むと例外が飛んでくるけど、漢は黙って無視するのがお作法らしい？"""
    try:
        win.addstr(y, x, s)
    except curses.error:
        pass


def human_play(stdscr):
    player = DebugHumanPlayer(stdscr)
    game = JumpGame(player)
    game.add_observer(player)
    player.setup_debug_screen(game)

    while True:
        game.play()

curses.wrapper(human_play)
