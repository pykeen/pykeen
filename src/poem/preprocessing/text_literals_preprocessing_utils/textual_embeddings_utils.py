# -*- coding: utf-8 -*-

"""Utils to process and embed text passages."""

import multiprocessing
from multiprocessing import Pool

import numpy as np
import spacy
from typing import Dict

nlp = spacy.load('en')


def _extract_sentences(descriptions: np.array) -> list:
    """Extract first n sentence of description."""
    processed_descriptions = []

    for description in descriptions:
        # Remove quotations marks from the beginning and the end
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]

        sentences = []
        # doc is a container
        doc = nlp(str(description))
        for i, sent in enumerate(doc.sents):
            sentences.append(sent.text)
            if i == MAX_NUM_SENTENCES - 1:
                break

        sentences = " ".join(sentences)
        processed_descriptions.append(sentences)

    return processed_descriptions


def extract_first_n_sentences(entity_to_desc: Dict[str, str],
                              max_num_sentences: int,
                              num_processes=multiprocessing.cpu_count()) -> Dict[str, str]:
    """Extract first n sentences of description, and return mapping of entity to processed descriptions."""
    global MAX_NUM_SENTENCES
    MAX_NUM_SENTENCES = max_num_sentences

    entities = list(entity_to_desc.keys())
    descriptions = list(entity_to_desc.values())
    description_chunks = np.array_split(descriptions, num_processes)

    with Pool(num_processes) as p:
        # Order of results is same as order description_chunks
        # results is a list of lists where each sublist represent the result of a process
        results = p.map(_extract_sentences, description_chunks)

    processed_decs = []

    for r in results:
        processed_decs += r

    entity_to_desc = dict(zip(entities, processed_decs))

    return entity_to_desc


def embedd_texts(texts:list) -> np.array:
    """"""
    pass