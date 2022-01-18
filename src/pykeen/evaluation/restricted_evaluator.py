"""Sampled evaluator from [teru2020]_."""
from collections import defaultdict
from typing import Dict, Tuple, Optional

import numpy as np
import torch

from .rank_based_evaluator import RankBasedEvaluator, compute_rank_from_scores, SIDE_HEAD, SIDE_TAIL
from ..triples import CoreTriplesFactory
from ..triples.triples_factory import MappedTriples


def sample_negatives(
    valid_triples: CoreTriplesFactory,
    all_pos: CoreTriplesFactory,
    num_samples: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:

    """
    Samples num_samples _filtered_ negative entities, ie, we make sure
    triples with sampled heads / tails do not exist in the input graph.
    For the inductive LP setup, the input graph is supposed to be inductive_inference graph,
    not transductive_training.

    Currently, the code is not very efficient in favor of replicating the original behavior

    :param valid_triples: CoreTriplesFactory
        Triples to be evaluated
    :param all_pos: CoreTriplesFactory
        The inference graph on which we run the valid_triples. Needed for filtering
    :param num_samples: int
        Number of random entities to sample
    """

    val_triples = valid_triples.mapped_triples
    all_pos_triples = all_pos.mapped_triples
    num_entities = all_pos.num_entities

    head_samples, tail_samples = [[] for _ in range(len(val_triples))], [[] for _ in range(len(val_triples))]

    # indexing positive triples
    head_index, tail_index = defaultdict(list), defaultdict(list)
    negs_dict = {"head": {}, "tail": {}}

    for triple in all_pos_triples:
        h, r, t = triple[0].item(), triple[1].item(), triple[2].item()
        tail_index[(h, r)].append(t)
        head_index[(r, t)].append(h)

    # sampling N negatives for each triple
    for i, row in enumerate(val_triples):
        head, rel, tail = row[0].item(), row[1].item(), row[2].item()

        head_samples[i].append(head)
        while len(head_samples[i]) < num_samples:

            neg_head = np.random.choice(num_entities)

            # make sure the triple with the sampled head is not a positive triple
            if neg_head != head and neg_head not in head_index[(rel, tail)]:
                head_samples[i].append(neg_head)

        # dumping to the index
        negs_dict["head"][(head, rel, tail)] = torch.tensor(head_samples[i], dtype=torch.long)

        tail_samples[i].append(tail)

        while len(tail_samples[i]) < num_samples:

            neg_tail = np.random.choice(num_entities)

            # make sure the triple with the sampled tail is not a positive triple
            if neg_tail != tail and neg_tail not in tail_index[(head, rel)]:
                tail_samples[i].append(neg_tail)

        # dumping to the index
        negs_dict["tail"][(head, rel, tail)] = torch.tensor(tail_samples[i], dtype=torch.long)

    head_samples = torch.tensor(head_samples, dtype=torch.long)
    tail_samples = torch.tensor(tail_samples, dtype=torch.long)

    return head_samples, tail_samples, negs_dict


def generate_dict(
        valid_triples: CoreTriplesFactory,
        head_samples: torch.Tensor,
        tail_samples: torch.Tensor
) -> Dict:
    """
    A small function to create a dictionary of triple -> negative entities for head and tail sides
    """
    val_triples = valid_triples.mapped_triples
    negs_dict = {}
    for triple, head_sample, tail_sample in zip(val_triples, head_samples, tail_samples):
        negs_dict["head"][tuple(triple.tolsit())] = head_sample
        negs_dict["tail"][tuple(triple.tolsit())] = tail_sample

    return negs_dict


class RestrictedRankBasedEvaluator(RankBasedEvaluator):
    def __init__(
        self,
        validation_factory: CoreTriplesFactory,
        all_pos: CoreTriplesFactory,
        num_negatives: int = 50,  # default for inductive lp by [teru2020]
        head_samples: torch.Tensor = None,  # shape: [num_valid_triples, n]
        tail_samples: torch.Tensor = None,  # shape: [num_valid_triples, n],
        **kwargs,
    ):
        """Restricted version of the rank-based evaluator.
        This evaluator can replicate the behavior of the inductive LP evaluator by [teru2020]
        where each validation / test triple is compared to only 50 randomly selected negatives.
        Original code: https://github.com/kkteru/grail/blob/2a3dffa719518e7e6250e355a2fb37cd932de91e/test_ranking.py#L75

        By default (if head_samples and tail_samples are not supplied), the sampling procedure
        generates 50 random negative entities _for each_ triple in the validation/test set.
        Therefore, the shape of head_samples and tail_samples will be [num_eval_triples, n].

        We don't use the built-in restrict_entities_to in the default evaluator since it selects the same K
        negatives for all evaluation triples. Instead, here each triple has its own negatives.

        :param validation_factory
            Triples to be evaluated at validation/test time
        :param all_pos
            All positive triples, the training graph for the transductive case,
            inductive_inference graph for the inductive case
        :param ks:
            The values for which to calculate hits@k. Defaults to {1,3,5,10}.
        :param filtered:
            Whether to use the filtered evaluation protocol. If enabled, ranking another true triple higher than the
            currently considered one will not decrease the score.
        :param num_negatives: dtype: int
            Number of negative entities (per triple) to sample. Defaults to 50 to replicate [teru2020]
        :param head_samples: shape: (num_valid_triples, n), dtype: torch.Tensor
            Integer ids of entities to be considered negative samples against which head prediction is evaluated.
            If None, the evaluator will generate them using the published procedure
        :param tail_samples: shape: (num_valid_triples, n), dtype: torch.Tensor
            Integer ids of entities to be considered negative samples against which tail prediction is evaluated.
            If None, the evaluator will generate them using the published procedure

        """
        super().__init__(**kwargs)

        if head_samples is None or tail_samples is None:
            self.head_samples, self.tail_samples, self.negs_dict = sample_negatives(
                valid_triples=validation_factory, all_pos=all_pos, num_samples=num_negatives
            )
        else:
            self.head_samples, self.tail_samples = head_samples, tail_samples
            self.negs_dict = generate_dict(validation_factory, head_samples, tail_samples)


    def _update_ranks_(
        self,
        true_scores: torch.FloatTensor,
        all_scores: torch.FloatTensor,
        side: str,
        triples: MappedTriples = None,
    ) -> None:
        """Shared code for updating the stored ranks for head/tail scores.

        :param true_scores: shape: (batch_size,)
        :param all_scores: shape: (batch_size, num_entities)
        :param side: head/tail
        :param triples: actual triples in a batch on which we are evaluating, needed for dict lookup
        """

        # land on a cpu for dictionary lookup
        triples = triples.cpu().tolist()
        sampled_entities = torch.stack([
            self.negs_dict[side][tuple(triple)] for triple in triples
        ], dim=0).to(all_scores.device)

        batch_ranks = compute_rank_from_scores(
            true_score=true_scores,
            all_scores=all_scores.gather(1, sampled_entities),
        )
        self.num_entities = all_scores.shape[1]
        for k, v in batch_ranks.items():
            self.ranks[side, k].extend(v.detach().cpu().tolist())

    def process_tail_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=SIDE_TAIL, triples=hrt_batch)

    def process_head_scores_(
        self,
        hrt_batch: MappedTriples,
        true_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
        dense_positive_mask: Optional[torch.FloatTensor] = None,
    ) -> None:  # noqa: D102
        self._update_ranks_(true_scores=true_scores, all_scores=scores, side=SIDE_HEAD, triples=hrt_batch)
