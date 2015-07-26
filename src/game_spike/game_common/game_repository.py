#!/usr/bin/env python
# coding: utf-8

import cPickle as pickle
import os
import bz2


class GameRepository(object):
    def __init__(self, base_dir=None):
        self.base_dir = os.path.abspath(base_dir or os.path.expanduser("~/.game"))
        self.model_dir = "%s/model" % self.base_dir
        self.safe_create_dir(self.base_dir)
        self.safe_create_dir(self.model_dir)

    @staticmethod
    def safe_create_dir(base_dir):
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

    def get_model_path(self, model_name):
        return "%s/%s/model.pkl" % (self.model_dir, model_name)

    def get_play_data_dir(self, model_name):
        return "%s/%s/play_data" % (self.model_dir, model_name)

    def get_play_data_index_path(self, model_name):
        return "%s/index.pkl" % (self.get_play_data_dir(model_name))

    def get_play_data_path_by_id(self, model_name, play_id):
        return "%s/%s" % (self.get_play_data_dir(model_name), play_id)

    def save_game_play(self, model_name, play_id, data):
        data_path = self.get_play_data_path_by_id(model_name, play_id)
        self._safe_save_data(data_path, data)
        # update index
        index_list = self.load_play_index(model_name)
        index_list.append(play_id)
        index_path = self.get_play_data_index_path(model_name)
        self._safe_save_data(index_path, index_list)

    def load_play_index(self, model_name):
        index_path = self.get_play_data_index_path(model_name)
        index_list = self._safe_load_data(index_path) or []
        return index_list

    def load_all_game_play(self, model_name):
        play_data_list = []
        index_list = self.load_play_index(model_name)
        for play_id in index_list:
            data_path = self.get_play_data_path_by_id(model_name, play_id)
            data = self._safe_load_data(data_path)
            play_data_list.append(data)
        return play_data_list

    def load_model_params(self, agent_model):
        """

        :type agent_model: AgentModel
        """
        model_path = self.get_model_path(agent_model.model_name)
        if os.path.exists(model_path):
            with file(model_path, "rb") as f:
                data = pickle.load(f)
                agent_model.function_set.parameters = data["parameters"]
                agent_model.set_extra_params(data.get("extra_params"))
                agent_model.meta = data["meta"]

    def save_model_params(self, agent_model):
        """

        :type agent_model: AgentModel
        """
        model_path = self.get_model_path(agent_model.model_name)
        data = {
            "parameters": agent_model.function_set.parameters,
            "extra_params": agent_model.get_extra_params(),
            "meta": agent_model.meta,
        }
        self._safe_save_data(model_path, data)

    def _safe_save_data(self, path, data):
        self.safe_create_dir(os.path.dirname(path))
        with file("%s.tmp" % path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename("%s.tmp" % path, path)

    def _safe_load_data(self, path):
        if os.path.exists(path):
            with file(path, "rb") as f:
                return pickle.load(f)
        return None
