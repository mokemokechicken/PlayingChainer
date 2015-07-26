# coding: utf-8

from cPickle import dumps, HIGHEST_PROTOCOL
import socket
import threading
import datetime

from game_repository import GameRepository
from replay_const import REPLAY_TYPE_CURRENT_PLAY, REPLAY_TYPE_HIGH_SCORES, REPLAY_TYPE_LAST_PLAY


class ReplayServer(object):
    last_play = None
    current_play = None
    MAX_SCENES = 10000

    info = ""
    player_name = "unknown"

    def __init__(self, port=None, host=None, repo=None):
        self.port = port or 7000
        self.host = host or '0.0.0.0'
        self.repo = repo or GameRepository()
        self.start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

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
        replay_mode = conn.recv(4096)
        if replay_mode == REPLAY_TYPE_CURRENT_PLAY:
            data = dumps(self.current_play, HIGHEST_PROTOCOL)
        elif replay_mode == REPLAY_TYPE_HIGH_SCORES:
            data = dumps(self.load_all_high_score_games(), HIGHEST_PROTOCOL)
        else:
            data = dumps(self.last_play,  HIGHEST_PROTOCOL)
        conn.send(data)

    def record_high_score_game(self, game):
        play_id = "%s-%s" % (self.start_time, game.play_id)
        self.repo.save_game_play(self.player_name, play_id, self.current_play)

    def load_all_high_score_games(self):
        play_data_list = self.repo.load_all_game_play(self.player_name)
        return play_data_list

    def on_game_start(self, game):
        self.player_name = game.player.name()
        self.current_play = {
            "size": [game.WIDTH, game.HEIGHT],
            "meta": game.meta_info(),
            "info": self.info,
            "scenes": [],
        }

    def on_game_over(self, game):
        self.last_play = self.current_play
        if game.high_score < game.total_reward:
            self.record_high_score_game(game)

    def on_update(self, game):
        if len(self.current_play["scenes"]) < self.MAX_SCENES:
            scene_data = {
                "screen": game.state.screen.data.copy(),
                "game": game.turn_info(),
            }
            if game.player.turn_info:
                scene_data["player"] = game.player.turn_info()
            self.current_play["scenes"].append(scene_data)
        else:
            self.last_play = self.current_play
