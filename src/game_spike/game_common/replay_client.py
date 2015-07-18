#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

from cPickle import loads
import curses
import socket
import time

from curse_util import ignore_error_add_str


def receive_all(sock):
    data = ""
    part = None
    while part != "":
        part = sock.recv(4096)
        data += part
    return data


class ReplayClient(object):
    W_TOP = 1
    W_LEFT = 1
    SEC_PER_TURN = 0.05

    main_window = None
    info_window = None

    def __init__(self, stdscr, host, port):
        self.stdscr = stdscr
        self.host = host
        self.port = port

    def poll(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        data = receive_all(sock)
        return loads(data)

    def replay_forever(self):
        while True:
            replay_data = self.poll()
            if replay_data is not None:
                self.play_data(replay_data)
            else:
                time.sleep(1)

    def init_window(self, width, height):
        curses.curs_set(0)
        border_win = curses.newwin(height + 2, width + 2, self.W_TOP, self.W_LEFT)  # h, w, y, x
        border_win.box()
        self.stdscr.refresh()
        border_win.refresh()
        self.main_window = curses.newwin(height, width, self.W_TOP + 1, self.W_LEFT + 1)
        self.main_window.refresh()
        self.info_window = curses.newwin(height, width, self.W_TOP + 1, self.W_LEFT + width + 2)

    def play_data(self, replay_data):
        width, height = replay_data["size"]
        meta_info = replay_data["meta"]
        if self.main_window is None:
            self.init_window(width, height)

        for scene in replay_data["scenes"]:
            t1 = time.time()
            game_info = scene["game"]
            game_info["screen_width"] = width
            game_info["screen_height"] = height
            screen = scene["screen"]
            self.update_screen(game_info, screen, meta_info)
            t2 = time.time() - t1
            if t2 < self.SEC_PER_TURN:
                time.sleep(self.SEC_PER_TURN - t2)

    def update_screen(self, game, screen, meta):
        for y in range(game["screen_height"]):
            line = "".join([chr(ch) for ch in screen[y]])
            ignore_error_add_str(self.main_window, y, 0, line)
        self.main_window.refresh()

        last_action = game["last_action"]
        keymap = meta["keymap"]
        play_id = meta["play_id"]

        self.info_window.clear()
        self.info_window.addstr(0, 2, "PlayID: %d    HighScore: %s" % (play_id, meta["high_score"]))
        self.info_window.addstr(1, 2, "Turn: %s" % game["turn"])
        self.info_window.addstr(2, 2, "Total Score: %s" % game["total_reward"])
        self.info_window.addstr(3, 2, "This Reward: %s" % game["last_reward"])
        self.info_window.addstr(4, 2, "=== Action ===")
        self.info_window.addstr(5, 2, "Left : %d" % (keymap["LEFT"] & last_action > 0))
        self.info_window.addstr(6, 2, "Right: %d" % (keymap["RIGHT"] & last_action > 0))
        self.info_window.addstr(7, 2, "Up   : %d" % (keymap["UP"] & last_action > 0))
        self.info_window.addstr(8, 2, "Down : %d" % (keymap["DOWN"] & last_action > 0))
        self.info_window.addstr(9, 2, "A    : %d" % (keymap["A"] & last_action > 0))
        self.info_window.addstr(10, 2, "B    : %d" % (keymap["B"] & last_action > 0))
        self.info_window.refresh()

def main(stdscr):
    client = ReplayClient(stdscr, 'localhost', 7000)
    client.replay_forever()


if __name__ == '__main__':
    curses.wrapper(main)
