# coding: utf-8
import curses

__author__ = 'k_morishita'


def ignore_error_add_str(win, y, x, s):
    """一番右下に書き込むと例外が飛んでくるけど、漢は黙って無視するのがお作法らしい？"""
    try:
        win.addstr(y, x, s)
    except curses.error:
        pass