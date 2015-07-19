# coding: utf-8

from cPickle import dumps, HIGHEST_PROTOCOL
import socket
import threading


class ReplayServer(object):
    last_play = None
    current_play = None
    MAX_SCENES = 10000

    info = ""

    def __init__(self, port=None, host=None):
        self.port = port or 7000
        self.host = host or '0.0.0.0'

    def run_as_background(self):
        server_thread = threading.Thread(target=self.serve_forever)
        server_thread.setDaemon(True)
        server_thread.start()

    def serve_forever(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((self.host, self.port))
            sock.listen(1)
            while True:
                conn, address = sock.accept()
                self.handle(conn)
                conn.close()
        finally:
            if sock:
                sock.close()

    def handle(self, conn):
        data = dumps(self.last_play,  HIGHEST_PROTOCOL)
        conn.send(data)

    def on_game_start(self, game):
        self.current_play = {
            "size": [game.WIDTH, game.HEIGHT],
            "meta": game.meta_info(),
            "info": self.info,
            "scenes": [],
        }

    def on_game_over(self, game):
        self.last_play = self.current_play

    def on_update(self, game):
        if len(self.current_play["scenes"]) < self.MAX_SCENES:
            self.current_play["scenes"].append({
                "screen": game.state.screen.data.copy(),
                "game": game.turn_info(),
            })
        else:
            self.last_play = self.current_play
