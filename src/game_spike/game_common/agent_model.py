# coding: utf-8
import datetime

__author__ = 'k_morishita'


class AgentModel(object):
    meta = {}

    def __init__(self, model, model_name, width, height, history_size, out_size):
        self.function_set = model
        self.model_name = model_name or ('created_%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.width = width
        self.height = height
        self.history_size = history_size
        self.in_size = width * height * self.history_size
        self.out_size = out_size
        self.meta['name'] = self.model_name

    def forward(self, in_variable):
        x = in_variable
        y = None
        for i in range(1, 1000):  # 1000 は適当な数
            if hasattr(self.function_set, "l%d" % i):
                x = getattr(self.function_set, "l%d" % i)(x)
            else:
                y = x
                break
        return y

    def on_learn(self, times):
        self.meta['learn_times'] = self.meta.get('learn_times', 0) + times

    def meta_str(self):
        return "name=%s learn_times=%s" % (self.model_name, self.meta.get('learn_times'))
