#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

from curse_util import ignore_error_add_str
import time
import curses

class DebugHumanPlayer(object):
    keymap = {}
    MSEC_PER_TURN = 100
    SEC_PER_TURN = MSEC_PER_TURN / 1000.0

    def __init__(self, stdscr):
        self.stdscr = stdscr

    def setup(self, game):
        self.stdscr.timeout(self.MSEC_PER_TURN - 5)
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
        c = self.stdscr.getch()
        t2 = time.time() - t1
        if t2 < self.SEC_PER_TURN:
            time.sleep(self.SEC_PER_TURN - t2)
        return self.keymap.get(c, 0)

class ConsoleDebugObserver(object):
    W_LEFT = 1
    W_TOP = 1
    INFO_WIDTH = 40
    MSEC_PER_TURN = 100
    SEC_PER_TURN = MSEC_PER_TURN / 1000.0

    main_window = None
    info_window = None

    def __init__(self, stdscr):
        self.stdscr = stdscr

    def setup(self, game):
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

    def on_game_start(self, game):
        pass

    def on_game_over(self, game):
        pass

    def on_update(self, game):
        screen = game.state.screen
        for y in range(game.HEIGHT):
            line = "".join([chr(ch) for ch in screen.data[y]])
            ignore_error_add_str(self.main_window, y, 0, line)
        self.main_window.refresh()

        self.info_window.clear()
        self.info_window.addstr(0 , 2, "PlayID: %d    HighScore: %s" % (game.play_id, game.high_score))
        self.info_window.addstr(1 , 2, "Turn: %s" % game.turn)
        self.info_window.addstr(2 , 2, "Total Score: %s" % game.total_reward)
        self.info_window.addstr(3 , 2, "This Reward: %s" % game.last_reward)
        self.info_window.addstr(4 , 2, "=== Action ===")
        self.info_window.addstr(5 , 2, "Left : %d" % (game.KEY_LEFT  & game.last_action > 0))
        self.info_window.addstr(6 , 2, "Right: %d" % (game.KEY_RIGHT & game.last_action > 0))
        self.info_window.addstr(7 , 2, "Up   : %d" % (game.KEY_UP    & game.last_action > 0))
        self.info_window.addstr(8 , 2, "Down : %d" % (game.KEY_DOWN  & game.last_action > 0))
        self.info_window.addstr(9 , 2, "A    : %d" % (game.BUTTON_A  & game.last_action > 0))
        self.info_window.addstr(10, 2, "B    : %d" % (game.BUTTON_B  & game.last_action > 0))
        self.info_window.refresh()
