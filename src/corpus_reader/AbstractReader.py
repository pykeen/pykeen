# -*- coding: utf-8 -*-add # -*- coding: utf-8 -*-
import abc

class AbstractReader(object):

    @abc.abstractmethod
    def retreive_knowledge_graph(self, enforce_download=False):
        return

    @abc.abstractmethod
    def transorm_corpus_to_id_format(temp_corpus, corpus_path):
        pass