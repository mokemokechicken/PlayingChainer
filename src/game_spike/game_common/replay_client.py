#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import os
from cPickle import loads
import curses
import socket
import time

from curse_util import ignore_error_add_str

from replay_const import REPLAY_TYPE_CURRENT_PLAY, REPLAY_TYPE_HIGH_SCORES, REPLAY_TYPE_LAST_PLAY


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
        self.replay_mode = REPLAY_TYPE_CURRENT_PLAY

    def poll(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        sock.send(self.replay_mode)
        sock.shutdown(1)  # close write socket
        data = receive_all(sock)
        sock.close()
        return loads(data)

    def replay_forever(self):
        while True:
            replay_data = self.poll()
            self.control_play_data(replay_data)

    def init_window(self, width, height):
        curses.curs_set(0)
        border_win = curses.newwin(height + 2, width + 2, self.W_TOP, self.W_LEFT)  # h, w, y, x
        border_win.box()
        self.stdscr.refresh()
        border_win.refresh()
        self.main_window = curses.newwin(height, width, self.W_TOP + 1, self.W_LEFT + 1)
        self.main_window.refresh()
        self.main_window.timeout(1)
        self.info_window = curses.newwin(20, 40, self.W_TOP + 1, self.W_LEFT + width + 2)

    def accept_change_mode(self):
        ch = self.main_window.getch()
        if ch == ord("1"):
            self.replay_mode = REPLAY_TYPE_LAST_PLAY
        elif ch == ord("2"):
            self.replay_mode = REPLAY_TYPE_CURRENT_PLAY
        elif ch == ord("3"):
            self.replay_mode = REPLAY_TYPE_HIGH_SCORES
        else:
            return False
        return True

    def control_play_data(self, replay_data):
        if replay_data is None:
            time.sleep(1)
            return

        if isinstance(replay_data, list):
            for r_data in replay_data:
                if not self.play_data(r_data):
                    return
        else:
            self.play_data(replay_data)

    def play_data(self, replay_data):
        width, height = replay_data["size"]
        meta_info = replay_data["meta"]
        if self.main_window is None:
            self.init_window(width, height)

        for scene in replay_data["scenes"]:
            t1 = time.time()
            if self.accept_change_mode():
                return False
            game_info = scene["game"]
            player_info = scene.get("player", {})
            game_info["screen_width"] = width
            game_info["screen_height"] = height
            screen = scene["screen"]
            self.update_screen(game_info, screen, player_info, meta_info, replay_data.get("info", []))
            t2 = time.time() - t1
            if t2 < self.SEC_PER_TURN:
                time.sleep(self.SEC_PER_TURN - t2)
        time.sleep(1)
        return True

    def update_screen(self, game, screen, player_info, meta, extra_info_list):
        for y in range(game["screen_height"]):
            line = "".join([chr(ch) for ch in screen[y]])
            ignore_error_add_str(self.main_window, y, 0, line)
        self.main_window.refresh()

        last_action = game["last_action"]
        keymap = meta["keymap"]
        play_id = meta["play_id"]

        self.info_window.clear()
        info_list = [
            "1: last, 2:current, 3:high",
            "REPLAY_MODE: %s" % self.replay_mode,
            "PlayID: %d    HighScore: %s" % (play_id, meta["high_score"]),
            "Turn: %s" % game["turn"],
            "Total Score: %s" % game["total_reward"],
            "This Reward: %s" % game["last_reward"],
            "LastLoss: %s" % round(player_info.get("loss_value") or -1, 7),
            "=== Action ===",
            "Left : %d" % (keymap["LEFT"] & last_action > 0),
            "Right: %d" % (keymap["RIGHT"] & last_action > 0),
            "Up   : %d" % (keymap["UP"] & last_action > 0),
            "Down : %d" % (keymap["DOWN"] & last_action > 0),
            "A    : %d" % (keymap["A"] & last_action > 0),
            "B    : %d" % (keymap["B"] & last_action > 0),
            "=== Info ===",
            ]
        info_list += extra_info_list
        self.write_info_lines(info_list)
        self.info_window.refresh()

    def write_info_lines(self, info_list):
        for i, info_str in enumerate(info_list):
            self.info_window.addstr(i, 2, info_str)

def main(stdscr):
    port = int(os.environ.get("GAME_SERVER_PORT", 7000))
    host = os.environ.get("GAME_SERVER_HOST", 'localhost')
    client = ReplayClient(stdscr, host, port)
    client.replay_forever()


if __name__ == '__main__':
    curses.wrapper(main)
