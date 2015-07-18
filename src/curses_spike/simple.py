#!/usr/bin/env python
# coding: utf-8

__author__ = 'k_morishita'

import curses


S_WIDTH = 40
S_HEIGHT = 24

def main(screen):
    curses.curs_set(0)
    border_win = curses.newwin(S_HEIGHT+2, S_WIDTH+2, 5, 5)  # h, w, y, x
    border_win.box()
    screen.refresh()
    border_win.refresh()
    #
    win = curses.newwin(S_HEIGHT, S_WIDTH, 6, 6)
    win.refresh()
    win.timeout(1000)
    x = 10
    y = 5
    key_x = {ord("l"): 1, ord("j"): -1}
    key_y = {ord("k"): 1, ord("i"): -1, ord("m"): 1}
    while True:
        c = win.getch()
        ignore_error_add_ch(win, y, x, ord(" "))
        x += key_x.get(c, 0)
        y += key_y.get(c, 0)
        x %= S_WIDTH
        y %= S_HEIGHT
        ignore_error_add_ch(win, y, x, ord("X"))
        win.refresh()

def ignore_error_add_ch(win, y, x, ch):
    try:
        win.addch(y, x, ch)
    except curses.error:
        pass




curses.wrapper(main)
