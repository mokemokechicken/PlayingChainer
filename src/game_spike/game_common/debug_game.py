#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import curses
import os

from debug_util import DebugHumanPlayer, ConsoleDebugObserver
from replay_server import ReplayServer

def human_play(stdscr, game_class):
    player = DebugHumanPlayer(stdscr)
    console = ConsoleDebugObserver(stdscr)
    replay_server = ReplayServer(int(os.environ.get("GAME_SERVER_PORT", 7000)))

    game = game_class(player)
    game.add_observer(console)
    game.add_observer(replay_server)

    player.setup(game)
    console.setup(game)
    replay_server.run_as_background()

    while True:
        game.play()

def debug_game(game_class):
    curses.wrapper(human_play, game_class)
