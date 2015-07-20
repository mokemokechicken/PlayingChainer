#!/usr/bin/env python
# coding: utf-8

import cPickle as pickle
import os

__author__ = 'k_morishita'


class GameRepository(object):
    def __init__(self, base_dir=None):
        self.base_dir = os.path.abspath(base_dir or os.path.expanduser("~/.game"))
        self.model_dir = "%s/model" % self.base_dir
        self.play_data_dir = "%s/play_data" % self.base_dir
        self.safe_create_dir(self.base_dir)
        self.safe_create_dir(self.model_dir)
        self.safe_create_dir(self.play_data_dir)

    @staticmethod
    def safe_create_dir(base_dir):
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir)

    def get_model_path(self, model_name):
        return "%s/%s.pkl" % (self.model_dir, model_name)

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
        with file("%s.tmp" % model_path, "wb") as f:
            data = {
                "parameters": agent_model.function_set.parameters,
                "extra_params": agent_model.get_extra_params(),
                "meta": agent_model.meta,
            }
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.rename("%s.tmp" % model_path, model_path)
