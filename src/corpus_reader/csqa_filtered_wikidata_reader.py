# -*- coding: utf-8 -*-
import json
import logging
import multiprocessing
import os
from multiprocessing import Pool

from utilities.constants import KG_EMBEDDINGS_PIPELINE_DIR, CSQA_WIKIDATA

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from corpus_reader.AbstractReader import AbstractReader



class CSQAWikiDataReader(AbstractReader):

    def _extract(self, argument_tuple):
        """

        :param argument_tuple:
        :return:
        """
        triples = []
        keys_of_kg_dict, kg = argument_tuple
        for subject in keys_of_kg_dict:
            value_dict = kg[subject]

            for predicate, object_list in value_dict.items():
                if not object_list:
                    continue
                for object in object_list:
                    triples.append((subject, predicate, object))

        return triples

    def _split_list_in_chunks(self, input_list, num_chunks):
        return [input_list[i::num_chunks] for i in range(num_chunks)]

    def transorm_corpus_to_id_format(self, temp_corpus, corpus_path):
        """

        :param temp_corpus:
        :param corpus_path:
        :return:
        """

        log.info("---------Load Corpus From Disk---------")
        with open(temp_corpus, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        keys = list(data.keys())
        num_processes = multiprocessing.cpu_count()
        log.info("Number of Used CPUs: ", num_processes)

        chunk_keys = self._split_list_in_chunks(input_list=keys, num_chunks=num_processes)
        chunksize = len(chunk_keys[0])

        log.info("---------Create Triples---------")
        with Pool(num_processes) as p:
            # triple_lists = p.map(self._extract, [(subset_keys, data) for subset_keys in chunk_keys])
            triple_lists = p.imap(self._extract, [(subset_keys, data) for subset_keys in chunk_keys],chunksize=chunksize)

        tripels = [item for sublist in triple_lists for item in sublist]

        log.info("---------Write Corpus---------")
        with open(corpus_path, 'w', encoding='utf-8') as f1:
            f1.write('@Comment@ Subject Predicate Object\n')
            for triple in tripels:
                f1.write('\t'.join(triple) + '\n')

        os.remove(temp_corpus)

    def retreive_knowledge_graph(self):
        """

        :return:
        """
        data_dir = os.path.join(KG_EMBEDDINGS_PIPELINE_DIR, CSQA_WIKIDATA)
        os.makedirs(data_dir, exist_ok=True)

        corpus_path = os.path.join(data_dir, "csqa_wikidata_short_1.csv")

        if not os.path.exists(corpus_path):
            temp_path = os.path.join(data_dir, "wikidata_short_1.json")
            self.transorm_corpus_to_id_format(temp_corpus=temp_path, corpus_path=corpus_path)

        return corpus_path
