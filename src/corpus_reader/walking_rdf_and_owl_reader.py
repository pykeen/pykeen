# -*- coding: utf-8 -*-
import logging
import os
from urllib.request import urlretrieve

from corpus_reader.AbstractReader import AbstractReader
from utilities.constants import KG_EMBEDDINGS_PIPELINE_DIR, WROC, WROC_URL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class WROCReader(AbstractReader):

    def transorm_corpus_to_id_format(self, temp_corpus, corpus_path):
        """

        :param temp_corpus:
        :param corpus_path:
        :return:
        """

        with open(temp_corpus, 'r', encoding='utf-8') as f1, open(corpus_path, 'w', encoding='utf-8') as f2:
            lines = f1.readlines()
            f2.write('# Subject Predicate Object \n')
            for line in lines:
                parts = line.strip().split(' ')
                parts = [p for p in parts if p != '' and p != '.']
                assert len(parts) == 3
                f2.write('\t'.join(parts) + '\n')

        os.remove(temp_corpus)

    def retreive_knowledge_graph(self, enforce_download=False):
        # Implementation based on https://github.com/bio2bel/hmdb/blob/master/src/bio2bel_hmdb/parser.py
        """

        :param enforce_download:
        :return:
        """
        data_dir = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, WROC)
        os.makedirs(data_dir, exist_ok=True)

        corpus_path = os.path.join(data_dir, "walking_rdf_and_owl.csv")

        if os.path.exists(corpus_path) and not enforce_download:
            log.info('Used existing corpus at %s', corpus_path)
        else:
            log.info('Download corpus')
            path, _ = urlretrieve(WROC_URL)
            self.transorm_corpus_to_id_format(temp_corpus=path, corpus_path=corpus_path)

        return corpus_path






