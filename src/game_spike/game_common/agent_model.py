# coding: utf-8

__author__ = 'k_morishita'


import datetime
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F

class AgentModel(object):
    meta = {}
    activate_functions = {}
    enable_gpu = False

    def __init__(self, model, model_name, width, height, history_size, out_size):
        self.function_set = model
        self.model_name = model_name or ('created_%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.width = width
        self.height = height
        self.history_size = history_size
        self.in_size = width * height * self.history_size
        self.out_size = out_size
        self.meta['name'] = self.model_name

    def setup_gpu(self):
        self.function_set.to_gpu()
        self.enable_gpu = True

    def forward(self, in_variable, train=True):
        x = in_variable
        if self.enable_gpu:
            x = cuda.to_gpu(x)
        y = None
        for i in range(1, 1000):  # 1000 は適当な数
            name = "l%d" % i
            if hasattr(self.function_set, name):
                li = getattr(self.function_set, name)
                if self.activate_functions.get(name):
                    func = self.activate_functions.get(name)
                    x = func(li(x), train=train)
                else:
                    x = li(x)
            else:
                y = x
                break
        return y

    def get_extra_params(self):
        return []

    def set_extra_params(self, params):
        pass

    def convert_state_to_input(self, state):
        return ((state.screen.data - 32) / 96.0).astype('float32')

    def on_learn(self, times):
        self.meta['learn_times'] = self.meta.get('learn_times', 0) + times

    def info_list(self):
        return [
            "name=%s" % self.model_name,
            "LearnTimes=%s" % self.meta.get('learn_times'),
        ]


class EmbedAgentModel(AgentModel):
    def __init__(self, embed_out_size, width, *args, **kw):
        super(EmbedAgentModel, self).__init__(width=width*embed_out_size, *args, **kw)
        self.embed_out_size = embed_out_size
        self._W = np.random.randn(96, embed_out_size).astype(np.float32)
        self._W[0] = np.zeros([embed_out_size], dtype=np.float32)

    def convert_state_to_input(self, state):
        return self._W[state.screen.data - 32].reshape(self.height, self.width)

    def get_extra_params(self):
        return [self._W]

    def set_extra_params(self, params):
        self._W = params[0]
