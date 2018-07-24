# -*- coding: utf-8 -*-
import json
import logging
import multiprocessing
import os
from urllib.request import urlretrieve
import zipfile


from utilities.constants import KG_EMBEDDINGS_PIPELINE_DIR, CSQA_WIKIDATA, WN_18, WN18_URL

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from corpus_reader.AbstractReader import AbstractReader


class WN18Reader(AbstractReader):

    def transorm_corpus_to_id_format(self, temp_corpus, corpus_path):
        """

        :param temp_corpus:
        :param corpus_path:
        :return:
        """

        data_dir = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, WN_18)
        os.makedirs(data_dir, exist_ok=True)
        zip_ref = zipfile.ZipFile(temp_corpus, 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        train_temp = os.path.join(data_dir,'WN18','train.txt')
        test_temp = os.path.join(data_dir,'test.txt')
        valid_temp = os.path.join(data_dir, 'valid.txt')

        train = os.path.join(data_dir,'wn_18_train.tsv')

        with open(train_temp, 'r', encoding='utf-8') as f1, open(train, 'w', encoding='utf-8') as f2:
            lines = f1.readlines()
            f2.write('@Comment@ Subject Predicate Object\n')
            for line in lines:
                parts = line.strip().split('\t')
                subject = parts[0]
                object = parts[1]
                predicate = parts[2]
                assert len(parts) == 3
                parts = [subject,predicate,object]
                f2.write('\t'.join(parts) + '\n')

        os.remove(temp_corpus)

    def retreive_knowledge_graph(self, enforce_download=False):
        """

        :return:
        """
        data_dir = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, WN_18)
        os.makedirs(data_dir, exist_ok=True)

        corpus_path = os.path.join(data_dir, "wn_18_train.tsv")

        if os.path.exists(corpus_path) and not enforce_download:
            log.info('Used existing corpus at %s', corpus_path)
        else:
            log.info('Download corpus')
            path, _ = urlretrieve(WN18_URL)
            self.transorm_corpus_to_id_format(temp_corpus=path, corpus_path=corpus_path)

        return corpus_path