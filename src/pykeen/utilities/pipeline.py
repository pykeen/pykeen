# -*- coding: utf-8 -*-

"""Implementation of the basic pipeline."""

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, Union

import torch
from sklearn.model_selection import train_test_split
from torch.nn import Module

from pykeen.constants import *
from pykeen.hyper_parameter_optimizer.random_search_optimizer import RandomSearchHPO
from pykeen.kge_models import get_kge_model
from pykeen.utilities.evaluation_utils.metrics_computations import compute_metric_results
from pykeen.utilities.train_utils import train_kge_model
from pykeen.utilities.triples_creation_utils import create_mapped_triples, create_mappings

__all__ = [
    'Pipeline',
]

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:  # TODO replace in Pipeline.run()
    """Results from the pipeline."""


class Pipeline(object):
    """"""

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.seed: int = config[SEED]
        self.entity_to_id: Dict[int: str] = None
        self.rel_to_id: Dict[int: str] = None
        self.device_name = (
            'cuda:0'
            if torch.cuda.is_available() and self.config[PREFERRED_DEVICE] == GPU else
            CPU
        )
        self.device = torch.device(self.device_name)

    @staticmethod
    def _use_hpo(config):
        return config[EXECUTION_MODE] == HPO_MODE

    @property
    def is_evaluation_required(self) -> bool:
        return TEST_SET_PATH in self.config or TEST_SET_RATIO in self.config

    def run(self) -> Mapping:
        """Run this pipeline."""
        if self._use_hpo(self.config):  # Hyper-parameter optimization mode
            mapped_pos_train_triples, mapped_pos_test_triples = self._get_train_and_test_triples()

            (trained_model,
             loss_per_epoch,
             entity_to_embedding,
             relation_to_embedding,
             eval_summary,
             params) = RandomSearchHPO.run(
                mapped_train_triples=mapped_pos_train_triples,
                mapped_test_triples=mapped_pos_test_triples,
                entity_to_id=self.entity_to_id,
                rel_to_id=self.rel_to_id,
                config=self.config,
                device=self.device,
                seed=self.seed,
            )
        else:  # Training Mode
            if self.is_evaluation_required:
                mapped_pos_train_triples, mapped_pos_test_triples = self._get_train_and_test_triples()
            else:
                mapped_pos_train_triples = self._get_train_triples()

            all_entities = np.array(list(self.entity_to_id.values()))

            # Initialize KG embedding model
            self.config[NUM_ENTITIES] = len(self.entity_to_id)
            self.config[NUM_RELATIONS] = len(self.rel_to_id)
            self.config[PREFERRED_DEVICE] = CPU if self.device_name == CPU else GPU
            kge_model: Module = get_kge_model(config=self.config)

            batch_size = self.config[BATCH_SIZE]
            num_epochs = self.config[NUM_EPOCHS]
            learning_rate = self.config[LEARNING_RATE]

            log.info("-------------Train KG Embeddings-------------")
            trained_model, loss_per_epoch = train_kge_model(
                kge_model=kge_model,
                all_entities=all_entities,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                batch_size=batch_size,
                pos_triples=mapped_pos_train_triples,
                device=self.device,
                seed=self.seed,
            )

            params = self.config
            eval_summary = None

            if self.is_evaluation_required:
                log.info("-------------Start Evaluation-------------")

                mean_rank, hits_at_k = compute_metric_results(
                    all_entities=all_entities,
                    kg_embedding_model=kge_model,
                    mapped_train_triples=mapped_pos_train_triples,
                    mapped_test_triples=mapped_pos_test_triples,
                    device=self.device,
                    filter_neg_triples=self.config[FILTER_NEG_TRIPLES],
                )

                eval_summary = OrderedDict()
                eval_summary[MEAN_RANK] = mean_rank
                eval_summary[HITS_AT_K] = hits_at_k

        # Prepare Output
        id_to_entity = {value: key for key, value in self.entity_to_id.items()}
        id_to_rel = {value: key for key, value in self.rel_to_id.items()}
        entity_to_embedding = {
            id_to_entity[id]: embedding.detach().cpu().numpy()
            for id, embedding in enumerate(trained_model.entity_embeddings.weight)
        }

        if self.config[KG_EMBEDDING_MODEL_NAME] in (SE_NAME, UM_NAME):
            relation_to_embedding = None
        else:
            relation_to_embedding = {
                id_to_rel[id]: embedding.detach().cpu().numpy()
                for id, embedding in enumerate(trained_model.relation_embeddings.weight)
            }

        return _make_results(
            trained_model=trained_model,
            loss_per_epoch=loss_per_epoch,
            entity_to_embedding=entity_to_embedding,
            relation_to_embedding=relation_to_embedding,
            eval_summary=eval_summary,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
            params=params,
        )

    def _get_train_and_test_triples(self) -> Tuple[np.ndarray, np.ndarray]:
        train_pos = load_data(self.config[TRAINING_SET_PATH])

        if TEST_SET_PATH in self.config:
            test_pos = load_data(self.config[TEST_SET_PATH])
        else:
            train_pos, test_pos = train_test_split(
                train_pos,
                test_size=self.config.get(TEST_SET_RATIO, 0.1),
                random_state=self.seed,
            )

        return self._handle_train_and_test(train_pos, test_pos)

    def _handle_train_and_test(self, train_pos, test_pos) -> Tuple[np.ndarray, np.ndarray]:
        """"""
        all_triples: np.ndarray = np.concatenate([train_pos, test_pos], axis=0)
        self.entity_to_id, self.rel_to_id = create_mappings(triples=all_triples)

        mapped_pos_train_triples, _, _ = create_mapped_triples(
            triples=train_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        mapped_pos_test_triples, _, _ = create_mapped_triples(
            triples=test_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        return mapped_pos_train_triples, mapped_pos_test_triples

    def _get_train_triples(self):
        train_pos = load_data(self.config[TRAINING_SET_PATH])

        self.entity_to_id, self.rel_to_id = create_mappings(triples=train_pos)

        mapped_pos_train_triples, _, _ = create_mapped_triples(
            triples=train_pos,
            entity_to_id=self.entity_to_id,
            rel_to_id=self.rel_to_id,
        )

        return mapped_pos_train_triples


def load_data(path: Union[str, Iterable[str]]) -> np.ndarray:
    """Load data given the *path*."""
    if isinstance(path, str):
        return _load_data_helper(path)

    return np.concatenate([
        _load_data_helper(p)
        for p in path
    ])


def _load_data_helper(path: str) -> np.ndarray:
    for prefix, handler in IMPORTERS.items():
        if path.startswith(f'{prefix}:'):
            return handler(path[len(f'{prefix}:'):])

    if path.endswith('.tsv'):
        return np.reshape(np.loadtxt(
            fname=path,
            dtype=str,
            comments='@Comment@ Subject Predicate Object',
            delimiter='\t',
        ), newshape=(-1, 3))

    if path.endswith('.nt'):
        import rdflib
        g = rdflib.Graph()
        g.parse(path, format='nt')
        return np.array(
            [
                [str(s), str(p), str(o)]
                for s, p, o in g
            ],
            dtype=np.str,
        )

    raise ValueError('''The argument to _load_data must be one of the following:
    
    - A string path to a .tsv file containing 3 columns corresponding to subject, predicate, and object
    - A string path to a .nt RDF file serialized in N-Triples format 
    - A string NDEx network UUID prefixed by "ndex:" like in ndex:f93f402c-86d4-11e7-a10d-0ac135e8bacf
    ''')


def _make_results(trained_model,
                  loss_per_epoch,
                  entity_to_embedding: Mapping[str, np.ndarray],
                  relation_to_embedding: Mapping[str, np.ndarray],
                  eval_summary,
                  entity_to_id,
                  rel_to_id,
                  params) -> Dict:
    results = OrderedDict()
    results[TRAINED_MODEL] = trained_model
    results[LOSSES] = loss_per_epoch
    results[ENTITY_TO_EMBEDDING]: Mapping[str, np.ndarray] = entity_to_embedding
    results[RELATION_TO_EMBEDDING]: Mapping[str, np.ndarray] = relation_to_embedding
    results[EVAL_SUMMARY] = eval_summary
    results[ENTITY_TO_ID] = entity_to_id
    results[RELATION_TO_ID] = rel_to_id
    results[FINAL_CONFIGURATION] = params
    return results
