# -*- coding: utf-8 -*-add # -*- coding: utf-8 -*-

import abc


class AbstractHPOptimizer(object):

    @abc.abstractmethod
    def optimize_hyperparams(self, config, path_to_kg, device, seed):
        return
