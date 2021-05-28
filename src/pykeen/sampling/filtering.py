# -*- coding: utf-8 -*-

r"""Consider the following properties of relation $r$. Because the corruption operations (see `Corruption`_)
are applied independently of triples, the resulting candidate corrupt triples could overlap with known positive
triples in $\mathcal{K}$.

=====================  ============================================  ==============================================================
Property of :math:`r`  Example pair of triples                       Implications
=====================  ============================================  ==============================================================
one-to-many            :math:`(h,r,t_1), (h,r,t_2) \in \mathcal{K}`  :math:`(h,r,t_2) \in T(h,r,t_1) \cup (h,r,t_1) \in T(h,r,t_2)`
multiple               :math:`(h,r_1,t), (h,r_2,t) \in \mathcal{K}`  :math:`(h,r_2,t) \in R(h,r_1,t) \cup (h,r_1,t) \in R(h,r_2,t)`
many-to-one            :math:`(h_1,r,t), (h_2,r,t) \in \mathcal{K}`  :math:`(h_2,r,t) \in H(h_1,r,t) \cup (h_1,r,t) \in H(h_2,r,t)`
=====================  ============================================  ==============================================================

If no relations in $\mathcal{K}$ satisfy any of the relevant properties for the corruption schema chosen in negative
sampling, then there is guaranteed to be no overlap between $\mathcal{N}$ and $\mathcal{K}$ such that 
$\mathcal{N} \cap \mathcal{K} \neq \emptyset$. However, this scenario is very unlikely for real-world knowledge graphs.

The known positive triples that appear in $\mathcal{N}$ are known false negatives. Hence, we know that these are 
incorrect (negative) training examples, and might want to exclude them to reduce the training noise.

.. warning:: 

    It should be taken into account that also a corrupted triple that is *not part*
    of the knowledge graph can represent a true fact. These "unknown" false negatives can
    not be removed *a priori* in the filtered setting. The philosophy of the methodology again relies
    on the low number of unknown false negatives such that learning can take place.


However, in practice, $|\mathcal{N}| \gg |\mathcal{K}|$, so the likelihood of generating a false negative is rather low. 
Therefore, the additional filter step is often omitted to lower computational cost. This general observation might not 
hold for all entities; e.g., for a hub entity which is connected to many other entities, there may be a considerable 
number of false negatives without filtering.
 

Identifying False Negatives During Training
-------------------------------------------
By default, PyKEEN does *not* filter false negatives from $\mathcal{N}$ during training. To enable filtering of
negative examples during training, the ``filtered`` keyword can be given to ``negative_sampler_kwargs`` like in:

.. code-block:: python

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            filtered=True,    
        ),
    )

PyKEEN implements several algorithms for filtering with different properties that can be chosen using the
``filterer`` keyword argument in ``negative_sampler_kwargs``. By default, an fast and approximate algorithm is used in
:class:`pykeen.sampling.filtering.BloomFilterer`, which is based on 
`bloom filters <https://en.wikipedia.org/wiki/Bloom_filter>`_. The bloom filterer also has a configurable desired error 
rate, which can be further lowered at the cost of increase in memory and computation costs. 

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            filtered=True,
            filterer='bloom',
            filterer_kwargs=dict(
                error_rate=0.0001,
            ),
        ),
    )

If you want to have a guarantee that all known false negatives are filtered, you can use a slower implementation based 
on Python's built-in sets, the :class:`pykeen.sampling.filtering.PythonSetFilterer`. It can be activated with:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        training_loop='sLCWA',
        negative_sampler='basic',
        negative_sampler_kwargs=dict(
            filtered=True,
            filterer='python-set',    
        ),
    )
    
Identifying False Negatives During Evaluation
---------------------------------------------
In contrast to training, PyKEEN **does** filter false negatives from $\mathcal{N}$ during evaluation by default.
To disable the "filtered setting" during evaluation, the ``filtered`` keyword can be given to ``evaluator_kwargs``
like in:

.. code-block:: python

    from pykeen.pipeline import pipeline

    results = pipeline(
        dataset='YAGO3-10',
        model='PairRE',
        evaluator_kwargs=dict(
            filtered=False,
        ),
    )

Filtering during evaluation is implemented differently than in negative sampling:

First, there are no choices between an exact or approximate algorithm via a
:class:`pykeen.sampling.filtering.Filterer`. Instead, the evaluation filtering can modify the
scores in-place and does so instead of selecting only the non-filtered entries. The reason is
mainly that evaluation always is done in 1:n scoring, and thus, we gain some efficiently here
by keeping the tensor in "dense" shape ``(batch_size, num_entities)``.

Second, filtering during evaluation has to be correct, and is crucial for reproducing results
from the filtered setting. For evaluation it makes sense to use all information we have to get
as solid evaluation results as possible.
"""  # noqa

import math
from abc import abstractmethod
from typing import Iterable

import torch
from class_resolver import Resolver
from torch import nn

from ..typing import MappedTriples

__all__ = [
    "filterer_resolver",
    "Filterer",
    "BloomFilterer",
    "PythonSetFilterer",
]


class Filterer(nn.Module):
    """An interface for filtering methods for negative triples."""

    def forward(
        self,
        negative_batch: MappedTriples,
    ) -> torch.BoolTensor:
        """Filter all proposed negative samples that are positive in the training dataset.

        Normally there is a low probability that proposed negative samples are positive in the training datasets and
        thus act as false negatives. This is expected to act as a kind of regularization, since it adds noise signal to
        the training data. However, the degree of regularization is hard to control since the added noise signal depends
        on the ratio of true triples for a given entity relation or entity entity pair. Therefore, the effects are hard
        to control and a researcher might want to exclude the possibility of having false negatives in the proposed
        negative triples.

        .. note ::
            Filtering is a very expensive task, since every proposed negative sample has to be checked against the
            entire training dataset.

        :param negative_batch: shape: (batch_size, num_negatives, 3)
            The batch of negative triples.

        :return: shape: (batch_size, num_negatives)
            A mask, where True indicates that the negative sample is valid.
        """
        return ~self.contains(batch=negative_batch)

    @abstractmethod
    def contains(self, batch: MappedTriples) -> torch.BoolTensor:
        """
        Check whether a triple is contained.

        Supports batching.

        :param batch: shape (batch_size, 3)
            The batch of triples.

        :return: shape: (batch_size,)
            Whether the triples are contained in the training triples.
        """
        raise NotImplementedError


class PythonSetFilterer(Filterer):
    """A filterer using Python sets for filtering.

    This filterer is expected to be rather slow due to the conversion from torch long tensors to Python tuples. It can
    still serve as a baseline for performance comparison.
    """

    def __init__(self, mapped_triples: MappedTriples):
        """Initialize the filterer.

        :param mapped_triples:
            The ID-based triples.
        """
        super().__init__()
        # store set of triples
        self.triples = set(map(tuple, mapped_triples.tolist()))

    def contains(self, batch: MappedTriples) -> torch.BoolTensor:  # noqa: D102
        return torch.as_tensor(
            data=[
                tuple(triple) in self.triples
                for triple in batch.view(-1, 3).tolist()
            ],
            dtype=torch.bool,
            device=batch.device,
        ).view(*batch.shape[:-1])


class BloomFilterer(Filterer):
    """
    A filterer for negative triples based on the Bloom filter.

    Pure PyTorch, a proper module which can be moved to GPU, and support batch-wise computation.

    .. seealso ::
        * https://github.com/hiway/python-bloom-filter/ - for calculation of sizes, and rough structure of code
        * https://github.com/skeeto/hash-prospector#two-round-functions - for parts of the hash function
    """

    #: some prime numbers for tuple hashing
    mersenne: torch.LongTensor

    #: The bit-array for the Bloom filter data structure
    bit_array: torch.BoolTensor

    def __init__(self, mapped_triples: MappedTriples, error_rate: float = 0.001):
        """
        Initialize the Bloom filter based filterer.

        :param mapped_triples:
            The ID-based triples.
        :param error_rate:
            The desired error rate.
        """
        super().__init__()

        # Allocate bit array
        self.ideal_num_elements = mapped_triples.shape[0]
        size = self.num_bits(num=self.ideal_num_elements, error_rate=error_rate)
        self.register_buffer(name="bit_array", tensor=torch.zeros(size, dtype=torch.bool))
        self.register_buffer(
            name="mersenne",
            tensor=torch.as_tensor(
                data=[2 ** x - 1 for x in [17, 19, 31]],
                dtype=torch.long,
            ).unsqueeze(dim=0),
        )

        # calculate number of hashing rounds
        self.rounds = self.num_probes(num_elements=self.ideal_num_elements, num_bits=size)

        # index triples
        self.add(triples=mapped_triples)

        # Store some meta-data
        self.error_rate = error_rate

    def __repr__(self):  # noqa:D105
        return (
            f"{self.__class__.__name__}("
            f"error_rate={self.error_rate}, "
            f"size={self.bit_array.shape[0]}, "
            f"rounds={self.rounds}, "
            f"ideal_num_elements={self.ideal_num_elements}, "
            f")"
        )

    @staticmethod
    def num_bits(num: int, error_rate: float = 0.01) -> int:
        """
        Determine the required number of bits.

        :param num:
            The number of elements the Bloom filter shall store.
        :param error_rate:
            The desired error rate.

        :return:
            The required number of bits.
        """
        numerator = -1 * num * math.log(error_rate)
        denominator = math.log(2) ** 2
        real_num_bits_m = numerator / denominator
        return int(math.ceil(real_num_bits_m))

    @staticmethod
    def num_probes(num_elements: int, num_bits: int):
        """
        Determine the number of probes / hashing rounds.

        :param num_elements:
            The number of elements.
        :param num_bits:
            The number of bits, i.e., the size of the Bloom filter.

        :return:
            The number of hashing rounds.
        """
        num_bits = num_bits
        real_num_probes_k = (num_bits / num_elements) * math.log(2)
        return int(math.ceil(real_num_probes_k))

    def probe(
        self,
        batch: MappedTriples,
    ) -> Iterable[torch.LongTensor]:
        """
        Iterate over indices from the probes.

        :param batch: shape: (batch_size, 3)
            A batch of elements.

        :yields: Indices of the k-th round, shape: (batch_size,).
        """
        # pre-hash
        x = (self.mersenne * batch).sum(dim=-1)
        for _ in range(self.rounds):
            # cf. https://github.com/skeeto/hash-prospector#two-round-functions
            x = x ^ (x >> 16)
            x = x * 0x7feb352d
            x = x ^ (x >> 15)
            x = x * 0x846ca68b
            x = x ^ (x >> 16)
            yield x % self.bit_array.shape[0]

    def add(self, triples: MappedTriples) -> None:
        """Add triples to the Bloom filter."""
        for i in self.probe(batch=triples):
            self.bit_array[i] = True

    def contains(self, batch: MappedTriples) -> torch.BoolTensor:
        """
        Check whether a triple is contained.

        :param batch: shape (batch_size, 3)
            The batch of triples.

        :return: shape: (batch_size,)
            The result. False guarantees that the element was not contained in the indexed triples. True can be
            erroneous.
        """
        result = batch.new_ones(batch.shape[:-1], dtype=torch.bool)
        for i in self.probe(batch):
            result &= self.bit_array[i]
        return result


filterer_resolver = Resolver.from_subclasses(
    base=Filterer,
    default=BloomFilterer,
)
